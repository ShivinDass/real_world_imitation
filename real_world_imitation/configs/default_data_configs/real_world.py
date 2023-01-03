from real_world_imitation.utils.general_utils import AttrDict
from real_world_imitation.data.real_world.src.real_data_loader import RealSequenceSplitDataset

data_spec = AttrDict(
    dataset_class=RealSequenceSplitDataset,
    n_actions=3,
    state_dim=2048*2+7+6+2, #4111=front_cam_emb(resnet18->512/resnet50->2048) + mount_cam_emb(resnet18->512/resnet50->2048) + ee_cartesian_pos_ob(3+4) + ee_cartesian_vel_ob(3+3) + gripper_pos[2]
    normalize=True,         #added
    use_goals=True, #added
)

data_spec.max_seq_len = 600
