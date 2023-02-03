import h5py
from collections import Counter
import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import ast

# transform = T.Compose(
#    [
#        T.Resize((224, 224)),  # 256 instead of 224 for vit-l
#        T.Normalize(
#            mean=[0.485, 0.456, 0.406],
#            std=[0.229, 0.224, 0.225],
#        ),
#    ]
# )
BATCH_SIZE = 1024


def process_data(file_paths, file_prefix):
    data_paths = [os.path.join(os.environ["DATA_DIR"], file_path) for file_path in file_paths]
    data = encode_trajectories(data_paths)
    data = remove_stationary_actions(data)
    data = remove_skipped_prompts_and_process_prompts(data)
    data = add_gripper_labels(data)
    data = split_into_primitive_trajectories(data)

    save_path = os.path.join(
        os.environ["SAVE_DIR"],
        "{}_embedded_r3m_clean_gripper_rnn_329".format(file_prefix),
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with h5py.File(
        os.path.join(save_path, "embedded_r3m_clean_gripper_rnn_329.hdf5"), "w"
    ) as g:
        for i in range(len(data["prompt_embeddings"])):
            grp = g.create_group(str(i))
            for k in data.keys():
                data[k][i] = np.array(data[k][i])
                grp.create_dataset(k, data=data[k][i])
    # print total size of data actions:
    print(
        "Total number of actions: {}".format(
            sum(len(data["actions"][i]) for i in range(len(data["actions"])))
        )
    )


def batch_process(model, tensor, device):
    outs = []
    for i in range(0, tensor.shape[0], BATCH_SIZE):
        feats = tensor[i : i + BATCH_SIZE].to(device)
        outs.append(model(feats))
    outs = torch.cat(outs)
    assert outs.shape[0] == tensor.shape[0]
    return outs


def encode_trajectories(data_dirs):

    from r3m import load_r3m

    device = "cuda" if torch.cuda.is_available() else "cpu"
    r3m = load_r3m("resnet50").eval().to(device)
    # mvp = mvp.load("vitb-mae-egosoup").to(device)
    # mvp.freeze()

    data_dict = {"prompts": []}
    demo_list = []
    for data_dir in data_dirs:
        demo_list.extend([os.path.join(data_dir, dd) for dd in os.listdir(data_dir)])
    demo_list = sorted(demo_list)
    print(demo_list)
    for g in demo_list:
        if os.path.isdir(g):
            continue
        print("==> Reading {}".format(g))

        with h5py.File(g, "r") as f:
            front_tensor = torch.tensor(
                np.array(f["front_cam_ob"]), dtype=torch.float32
            ).movedim(3, 1)
            mount_tensor = torch.tensor(
                np.array(f["mount_cam_ob"]), dtype=torch.float32
            ).movedim(3, 1)
            with torch.cuda.amp.autocast(enabled=True):
                with torch.no_grad():
                    f_embedding = batch_process(r3m, front_tensor, device)
                    m_embedding = batch_process(r3m, mount_tensor, device)

            data_dict["front_cam_emb"] = (
                np.concatenate(
                    (data_dict["front_cam_emb"], f_embedding.data.cpu().numpy().copy()),
                    axis=0,
                )
                if "front_cam_emb" in data_dict.keys()
                else f_embedding.data.cpu().numpy().copy()
            )
            data_dict["mount_cam_emb"] = (
                np.concatenate(
                    (data_dict["mount_cam_emb"], m_embedding.data.cpu().numpy().copy()),
                    axis=0,
                )
                if "mount_cam_emb" in data_dict.keys()
                else m_embedding.data.cpu().numpy().copy()
            )

            for k in f.keys():
                if k in [
                    "actions",
                    "ee_cartesian_pos_ob",
                    "ee_cartesian_vel_ob",
                    "joint_pos_ob",
                    "joint_vel_ob",
                    "terminals",
                ]:
                    data_dict[k] = (
                        np.concatenate((data_dict[k], np.array(f[k]).copy()), axis=0)
                        if k in data_dict.keys()
                        else np.array(f[k]).copy()
                    )
                elif k == "prompts":
                    data_dict[k].extend([prompt for prompt in f[k]])

            data_dict["terminals"][-1] = 1
            if "reward" in f.keys():
                a = np.array(f["reward"]).copy()
                data_dict["reward"] = (
                    np.concatenate((data_dict["reward"], a), axis=0)
                    if "reward" in data_dict.keys()
                    else a
                )
            else:
                a = np.zeros(len(np.array(f["actions"])))
                a[-2] = 1
                data_dict["reward"] = (
                    np.concatenate((data_dict["reward"], a), axis=0)
                    if "reward" in data_dict.keys()
                    else a
                )

    return data_dict


def remove_stationary_actions(f):
    clean_data = {k: [] for k in f}
    for i, a in enumerate(f["actions"]):
        if f["terminals"][i] == False and np.linalg.norm(f["actions"][i]) == 0:
            continue
        for k in f:
            # print(k, print(f[k].shape))
            clean_data[k].append(f[k][i])

    clean_data = {k: np.array(clean_data[k]) for k in clean_data.keys()}
    return clean_data


def split_into_primitive_trajectories(f):
    # rewarded and terminal if prompt is different from next prompt
    primitive_trajs_dict = {k: [[]] for k in f}
    for i in range(len(f["prompts"])):
        reward = 0
        new_traj = False
        if i == len(f["prompts"]) - 1:
            reward = 1
        elif f["prompts"][i] != f["prompts"][i + 1]:
            reward = 1
            new_traj = True
        f["reward"][i] = f["terminals"][i] = reward
        for k in f:
            primitive_trajs_dict[k][-1].append(f[k][i])
            if new_traj:
                primitive_trajs_dict[k].append([])

    # clear out length 1 trajectories
    prompts = set(f.pop("prompts"))
    prompts_list = primitive_trajs_dict.pop("prompts")
    # because there's a prompt for each traj we only need the first timestep
    prompts_list = [l[0] for l in prompts_list]
    print(prompts)
    # Count distribution of lengths
    lengths = [
        len(primitive_trajs_dict["reward"][i])
        for i in range(len(primitive_trajs_dict["reward"]))
    ]
    print(f"Length distribution: {Counter(lengths)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sentence_embedder = SentenceTransformer("all-MiniLM-L12-v2", device=device)
    print("Encoding language prompts now")
    with torch.no_grad():
        encoded_prompts = sentence_embedder.encode(
            prompts_list,
            #batch_size=128,
            show_progress_bar=True,
            #output_value="token_embeddings",
        )
        #encoded_prompts = [prompt.cpu().numpy() for prompt in encoded_prompts]

    # concat all the trajectories
    #for k in primitive_trajs_dict:
    #    primitive_trajs_dict[k] = np.concatenate(
    #        [primitive_trajs_dict[k][i] for i in range(len(primitive_trajs_dict[k]))]
    #    )
    # set the prompt embeddings
    primitive_trajs_dict["prompt_embeddings"] = encoded_prompts

    # for k in primitive_trajs_dict:
    #    primitive_trajs_dict[k] = [
    #        np.array(primitive_trajs_dict[k][i])
    #        for i in range(len(primitive_trajs_dict[k]))
    #        if len(primitive_trajs_dict[k][i]) > 1
    #    ]

    # create new key and value of timesteps until end of trajectory
    primitive_trajs_dict["steps_to_end"] = [
        np.arange(len(primitive_trajs_dict["reward"][i]) - 1, -1, -1)
        for i in range(len(primitive_trajs_dict["reward"]))
    ]

    return primitive_trajs_dict


def remove_skipped_prompts_and_process_prompts(f):
    clean_data = {k: [] for k in f}
    for i, p in enumerate(f["prompts"]):
        prompt = p.decode("utf-8")
        if f["terminals"][i] == False:
            if prompt == "skipped":
                continue
            elif ast.literal_eval(prompt)[0] == "go_to":
                continue
            elif ast.literal_eval(prompt)[0] == "slide_to":
                continue
            if prompt != "skipped":
                # process the prompt here
                tuple_prompt = ast.literal_eval(prompt)
                if tuple_prompt[0] == "pick":
                    object_name = " ".join(tuple_prompt[1][0].split("_"))
                    new_prompt = f"Pick up the {object_name}."
                elif tuple_prompt[0] == "place":
                    place_object_name = " ".join(tuple_prompt[1][0].split("_"))
                    new_prompt = f"Place the {object_name} in the {place_object_name}."
                else:
                    raise ValueError(f"Prompt not accounted for: {tuple_prompt[0]}")
        for k in f:
            # print(k, print(f[k].shape))
            if k == "prompts":
                clean_data[k].append(new_prompt)
            else:
                clean_data[k].append(f[k][i])

    clean_data = {k: np.array(clean_data[k]) for k in clean_data.keys()}
    return clean_data

def add_gripper_labels(f):
    labelled_data = {k: [] for k in f.keys()}
    for i, a in enumerate(f["actions"]):
        for k in f.keys():
            if not k == "actions":
                labelled_data[k].append(f[k][i])
        if a[-1] < 0:
            label = 0
        elif a[-1] == 0:
            label = 1
        else:
            label = 2

        labelled_data["actions"].append(
            np.concatenate((f["actions"][i][:-2], [label]), axis=0)
        )
    return labelled_data


if __name__ == "__main__":
    process_data(
        file_paths=["bootstrap_focused_data", "bootstrap_focused_data_2", "play_data", "play_data_2"], file_prefix="real_robot",
    )