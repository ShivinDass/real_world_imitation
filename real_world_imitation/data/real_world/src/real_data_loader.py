import numpy as np
import itertools
import h5py
import pickle
import os

from real_world_imitation.components.data_loader import Dataset
from real_world_imitation.utils.general_utils import AttrDict


class RealSequenceSplitDataset(Dataset):
    SPLIT = AttrDict(train=1, val=0.0, test=0.0)

    def __init__(self, data_dir, data_conf, phase, resolution=None, shuffle=True, dataset_size=-1):
        self.phase = phase
        self.data_dir = data_dir
        self.spec = data_conf.dataset_spec
        self.subseq_len = self.spec.subseq_len
        self.remove_goal = self.spec.remove_goal if 'remove_goal' in self.spec else False
        self.dataset_size = dataset_size
        self.device = data_conf.device
        self.n_worker = 4
        self.shuffle = shuffle
        
        self.dataset = self.get_dataset()

        # normalize inputs with mean 0 and variance 1 for each dimension
        if 'normalize' in self.spec and self.spec.normalize:
            self.input_mean = self.dataset['observations'][:, -15:].mean(axis=0)
            self.input_stddev = self.dataset['observations'][:, -15:].std(axis=0)
            self.dataset['observations'][:, -15:] = (self.dataset['observations'][:, -15:] - self.input_mean)/self.input_stddev

            self.output_mean = self.dataset['actions'][:,:3].mean(axis=0)
            self.output_stddev = self.dataset['actions'][:,:3].std(axis=0)
            self.dataset['actions'][:,:3] = (self.dataset['actions'][:,:3] - self.output_mean)/self.output_stddev

            #save mean and stddev to use during inference
            norm_file_path = os.path.join(os.path.dirname(self.data_dir), os.path.splitext(os.path.basename(self.data_dir))[0] + "_norm_constants.pkl")
            
            print("==> Normalizing input/output and saving mean and stddev to {}".format(norm_file_path))    
            with open(norm_file_path, 'wb') as f:
                pickle.dump(AttrDict(
                    # observations_mean=self.input_mean,
                    # observations_stddev=self.input_stddev,
                    actions_mean=self.output_mean,
                    actions_stddev=self.output_stddev
                ), f)
                print("Normalizing factors:", self.output_mean, self.output_stddev)

        # split dataset into sequences
        self.seq_end_idxs = np.where(self.dataset['terminals'])[0]
        start = 0
        self.seqs = []
        for end_idx in self.seq_end_idxs:
            if end_idx+1 - start < self.subseq_len: continue    # skip too short demos
            self.seqs.append(AttrDict(
                states=np.asarray(self.dataset['observations'][start:end_idx+1], dtype=np.float32),
                actions=np.asarray(self.dataset['actions'][start:end_idx+1], dtype=np.float32),
                rewards=np.asarray(self.dataset['reward'][start:end_idx+1], dtype=np.float32) if 'reward' in self.dataset.keys() else np.zeros((end_idx+1 - start,), dtype=np.float32),
                done=np.zeros((end_idx + 1 - start,), dtype=np.float32),
            ))
            if 'reward' not in self.dataset.keys():
                self.seqs[-1].rewards[-2] = 1.0
            self.seqs[-1].done[-2] = 1.0
            start = end_idx+1

        # 0-pad sequences for skill-conditioned training
        if 'pad_n_steps' in self.spec and self.spec.pad_n_steps > 0:
            for seq in self.seqs:
                seq.states = np.concatenate((np.zeros((self.spec.pad_n_steps, seq.states.shape[1]), dtype=seq.states.dtype), seq.states))
                seq.actions = np.concatenate((np.zeros((self.spec.pad_n_steps, seq.actions.shape[1]), dtype=seq.actions.dtype), seq.actions))

        # filter demonstration sequences
        if 'filter_indices' in self.spec:
            print("!!! Filtering kitchen demos in range {} !!!".format(self.spec.filter_indices))
            if not isinstance(self.spec.filter_indices[0], list):
                self.spec.filter_indices = [self.spec.filter_indices]
            self.seqs = list(itertools.chain.from_iterable([\
                list(itertools.chain.from_iterable(itertools.repeat(x, self.spec.demo_repeats)
                               for x in self.seqs[fi[0] : fi[1]+1])) for fi in self.spec.filter_indices]))
        import random
        random.Random(0).shuffle(self.seqs)
        self.n_seqs = len(self.seqs)

        if self.phase == "train":
            self.start = 0
            self.end = int(self.SPLIT.train * self.n_seqs)
        elif self.phase == "val":
            self.start = int(self.SPLIT.train * self.n_seqs)
            self.end = int((self.SPLIT.train + self.SPLIT.val) * self.n_seqs)
        else:
            self.start = int((self.SPLIT.train + self.SPLIT.val) * self.n_seqs)
            self.end = self.n_seqs

    def __getitem__(self, index):
        # sample start index in data range
        seq = self._sample_seq()
        start_idx = np.random.randint(0, seq.states.shape[0] - self.subseq_len + 1)
        output = AttrDict(
            states=seq.states[start_idx:start_idx+self.subseq_len],
            actions=seq.actions[start_idx:start_idx+self.subseq_len],
            pad_mask=np.ones((self.subseq_len,)),
            rewards=seq.rewards[start_idx:start_idx+self.subseq_len],
            done=seq.done[start_idx:start_idx + self.subseq_len],
        )

        # can make it faster by vectorizing the for loop
        if self.spec.use_goals:
            if isinstance(self.spec.goal_len, int):
                goals = np.stack([seq.states[min(seq.states.shape[0]-1, start_idx + self.spec.goal_len + i)] for i in range(self.subseq_len)])
            elif isinstance(self.spec.goal_len, (tuple, list)) and len(self.spec.goal_len) == 2:
                goals = np.stack([seq.states[min(seq.states.shape[0]-1, start_idx + np.random.random_integers(*self.spec.goal_len) + i)] for i in range(self.subseq_len)])
            else:
                raise "ValueError: goal_len incorrectly initialized"
            output.goals = goals

        if self.remove_goal:
            output.states = output.states[..., :int(output.states.shape[-1]/2)]
        return output

    def _sample_seq(self):
        return np.random.choice(self.seqs[self.start:self.end])

    def __len__(self):
        if self.dataset_size != -1:
            return self.dataset_size
        return int(self.SPLIT[self.phase] * self.dataset['observations'].shape[0] / self.subseq_len)

    def get_dataset(self):
        if self.data_dir is None:
            raise Exception(
                "Dataset directory not provided"
            )
        
        dataset_file = h5py.File(self.data_dir, 'r')
        data_dict = {k: np.array(dataset_file[k][:]) for k in dataset_file}
        dataset_file.close()

        data_dict['observations'] = data_dict['front_cam_emb'].copy()
        data_dict['observations'] = np.concatenate((data_dict['observations'], data_dict['prompt_embeddings']), axis=1)
        # for obs_key in ['mount_cam_emb', 'prompt_embeddings', 'ee_cartesian_pos_ob', 'ee_cartesian_vel_ob']:
        # for obs_key in ['mount_cam_emb', 'ee_cartesian_pos_ob', 'ee_cartesian_vel_ob']:
        #     data_dict['observations'] = np.concatenate((data_dict['observations'], data_dict[obs_key]), axis=1)
        
        # data_dict['observations'] = np.concatenate((data_dict['observations'], data_dict['joint_pos_ob'][:, -2:]), axis=1)
        
        for key in ['observations', 'actions', 'terminals']:
            assert key in data_dict, "Dataset is missing key %s" % key
            
        return data_dict
        
if __name__=='__main__':
    from real_world_imitation.configs.default_data_configs.real_world import data_spec

    data_conf = AttrDict()
    data_conf.dataset_spec = data_spec
    data_conf.dataset_spec.subseq_len = 15
    data_conf.device = 'cpu'
    a = RealSequenceSplitDataset(data_dir=os.path.join(os.environ['DATA_DIR'], "thrifty_all_new_tasks_embedded50_clean_gripper/embedded50_clean_gripper.hdf5"),
                                data_conf=data_conf, phase='train')
    for k in a.dataset.keys():
        print(k, a.dataset[k].shape)

    print(a.start, a.end)
    print(a.n_seqs)
    print(len(a.seq_end_idxs), a.seq_end_idxs, len(a))
    
    c1 = c2 = 0
    for i, e in enumerate(a.seq_end_idxs):
        if a.seqs[i].rewards[-2]==1:
            c1+=1
        if a.seqs[i].rewards[-1]==1:
            c2+=1

    print(len(a.seqs), c1, c2)