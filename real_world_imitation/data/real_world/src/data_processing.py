import h5py
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as T
import torch
import random

def process_data(file_path, file_prefix):
    data_path = os.path.join(os.environ['DATA_DIR'], file_path)
    data = encode_trajectories(data_path)
    data = remove_stationary_actions(data)
    data = add_gripper_labels(data)

    save_path = os.path.join(os.environ['DATA_DIR'], '{}_embedded50_clean_gripper'.format(file_prefix))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with h5py.File(os.path.join(save_path, 'embedded50_clean_gripper.hdf5'), 'w') as g:
        for k in data.keys():
            data[k] = np.array(data[k])
            print(k, data[k].shape)
            g.create_dataset(k, data=data[k])
    

def encode_trajectories(data_dir):

    from r3m import load_r3m
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    r3m = load_r3m('resnet50').eval().to(device)

    data_dict = {}
    demo_list = sorted(os.listdir(data_dir))
    for g in demo_list:
        g = os.path.join(data_dir, g)
        if os.path.isdir(g):
            continue
        print('==> Reading {}'.format(g))

        with h5py.File(g, 'r') as f:
            front_tensor = torch.tensor(np.array(f['front_cam_ob'])).movedim(3,1)
            mount_tensor = torch.tensor(np.array(f['mount_cam_ob'])).movedim(3,1)

            with torch.no_grad():                            
                f_embedding = r3m(front_tensor)
                m_embedding = r3m(mount_tensor)
            
            data_dict['front_cam_emb'] = np.concatenate((data_dict['front_cam_emb'], f_embedding.data.cpu().numpy().copy()), axis=0) if 'front_cam_emb' in data_dict.keys() else f_embedding.data.cpu().numpy().copy()
            data_dict['mount_cam_emb'] = np.concatenate((data_dict['mount_cam_emb'], m_embedding.data.cpu().numpy().copy()), axis=0) if 'mount_cam_emb' in data_dict.keys() else m_embedding.data.cpu().numpy().copy()

            for k in f.keys():
                if k in ['actions', 'ee_cartesian_pos_ob', 'ee_cartesian_vel_ob', 'joint_pos_ob', 'joint_vel_ob', 'terminals']:
                    data_dict[k] = np.concatenate((data_dict[k], np.array(f[k]).copy()), axis=0) if k in data_dict.keys() else np.array(f[k]).copy()

            data_dict['terminals'][-1] = 1
            if 'reward' in f.keys():
                a = np.array(f['reward']).copy()
                data_dict['reward'] = np.concatenate((data_dict['reward'], a), axis=0) if 'reward' in data_dict.keys() else a
            else:
                a = np.zeros(len(np.array(f['actions'])))
                a[-2] = 1
                data_dict['reward'] = np.concatenate((data_dict['reward'], a), axis=0) if 'reward' in data_dict.keys() else a

    return data_dict

def remove_stationary_actions(f):
    clean_data = {k:[] for k in f}
    for i, a in enumerate(f['actions']):
        if f['terminals'][i]==False and np.linalg.norm(f['actions'][i])==0 and f['reward'][i]==0:
            print(f['reward'][i])
            print(f['actions'][i])
            continue
        for k in f:
            # print(k, print(f[k].shape))
            clean_data[k].append(f[k][i])
    
    clean_data = {k: np.array(clean_data[k]) for k in clean_data.keys()}
    return clean_data

def add_gripper_labels(f):
    labelled_data = {k:[] for k in f.keys()}
    for i, a in enumerate(f['actions']):
        for k in f.keys():
            if not k=='actions':
                labelled_data[k].append(f[k][i])
        if a[-1]<0:
            label=0
        elif a[-1]==0:
            label=1
        else:
            label=2
        
        labelled_data['actions'].append(np.concatenate((f['actions'][i][:-2],[label]), axis=0))

    return labelled_data

if __name__=='__main__':
    process_data(file_path='boosting/boosted_datasets/jaco_nonzero/data/', file_prefix='boosting_boosted')
    
