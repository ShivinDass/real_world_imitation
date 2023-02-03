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

def process_data(file_path, file_prefix):
    data_path = os.path.join(os.environ["DATA_DIR"], file_path)
    data = encode_trajectories(data_path)
    data = remove_stationary_actions(data)
    data = remove_skipped_prompts_and_process_prompts(data)
    data = add_gripper_labels(data)
    encode_prompts(data)
    data = split_into_primitive_trajectories(data)
    data = filter_based_on_prompt(data, prompt="Pick up the apple fruit.")

    save_path = os.path.join(
        os.environ["DATA_DIR"], "{}_embedded50_clean_gripper".format(file_prefix)
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with h5py.File(
        os.path.join(save_path, "embedded50_clean_gripper.hdf5"), "w"
    ) as g:
        for k in data.keys():
            data[k] = np.array(data[k])
            print(k, data[k].shape)
            if k == 'prompts':
                continue
            g.create_dataset(k, data=np.array(data[k]))


def batch_process(model, tensor, device):
    outs = []
    for i in range(0, tensor.shape[0], BATCH_SIZE):
        feats = tensor[i : i + BATCH_SIZE].to(device)
        outs.append(model(feats))
    outs = torch.cat(outs)
    assert outs.shape[0] == tensor.shape[0]
    return outs


def encode_trajectories(data_dir):

    from r3m import load_r3m

    device = "cuda" if torch.cuda.is_available() else "cpu"
    r3m = load_r3m("resnet50").eval().to(device)
    # mvp = mvp.load("vitb-mae-egosoup").to(device)
    # mvp.freeze()

    data_dict = {"prompts": []}
    demo_list = sorted(os.listdir(data_dir))
    for g in demo_list:
        g = os.path.join(data_dir, g)
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
        terminal = 0
        new_traj = False
        if i == len(f["prompts"]) - 1:
            terminal = 1
            # reward = 1
        elif f["prompts"][i] != f["prompts"][i + 1] or f["terminals"][i]==1:
            terminal = 1
            # reward = 1
            new_traj = True
        f["reward"][i] = 0
        f["terminals"][i] = terminal
        for k in f:
            primitive_trajs_dict[k][-1].append(f[k][i])
            if new_traj:
                primitive_trajs_dict[k].append([])
    # clear out length 1 trajectories
    prompts = set(f["prompts"])
    # primitive_trajs_dict.pop("prompts")
    print(prompts)
    print(
        f"BEFORE CLEARING OUT LENGTH 1 TRAJECTORIES: {len(primitive_trajs_dict['reward'])}"
    )
    # Count distribution of lengths
    lengths = [
        len(primitive_trajs_dict["reward"][i])
        for i in range(len(primitive_trajs_dict["reward"]))
    ]
    print(f"Length distribution: {Counter(lengths)}")

    for k in primitive_trajs_dict:
        primitive_trajs_dict[k] = [
            np.array(primitive_trajs_dict[k][i])
            for i in range(len(primitive_trajs_dict[k]))
            if len(primitive_trajs_dict[k][i]) > 1
        ]
    print(
        f"AFTER CLEARING OUT LENGTH 1 TRAJECTORIES: {len(primitive_trajs_dict['reward'])}"
    )
    # create new key and value of timesteps until end of trajectory
    primitive_trajs_dict["steps_to_end"] = [
        np.arange(len(primitive_trajs_dict["reward"][i]) - 1, -1, -1)
        for i in range(len(primitive_trajs_dict["reward"]))
    ]

    # concat all the trajectories
    for k in primitive_trajs_dict:
        primitive_trajs_dict[k] = np.concatenate(
            [primitive_trajs_dict[k][i] for i in range(len(primitive_trajs_dict[k]))]
        )
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

def filter_based_on_prompt(f, prompt):
    filtered_data = {k: [] for k in f.keys()}
    for i, a in enumerate(f["actions"]):
        if f['prompts'][i] == 'Pick up the apple fruit.':
            print(f['prompts'][i])
            for k in f.keys():
                filtered_data[k].append(f[k][i])
        
    return filtered_data

def encode_prompts(f):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sentence_embedder = SentenceTransformer("all-MiniLM-L12-v2", device=device)
    all_prompts = [prompt for prompt in f["prompts"]]
    language_embeddings = []
    print("Encoding language prompts now")
    encoded_prompts = sentence_embedder.encode(
        all_prompts, batch_size=256, show_progress_bar=True
    )
    for i in range(encoded_prompts.shape[0]):
        language_embedding = encoded_prompts[i]
        language_embeddings.append(language_embedding)
    f["prompt_embeddings"] = np.stack(language_embeddings)

if __name__ == "__main__":
    process_data(
        file_path="play_data", file_prefix="play_data_pick_apple_only",
    )