import os
from real_world_imitation.models.bc_recurrent_mdl import BCRecurrentMdl
from real_world_imitation.components.logger import Logger
from real_world_imitation.utils.general_utils import AttrDict
from real_world_imitation.configs.default_data_configs.real_world import data_spec
from real_world_imitation.components.evaluator import DummyEvaluator


current_dir = os.path.dirname(os.path.realpath(__file__))


configuration = {
    'model': BCRecurrentMdl,
    'logger': Logger,
    'data_dir': os.path.join(os.environ['DATA_DIR'], 'all_play_data_diverse_mvp_clean_gripper/mvp_clean_gripper.hdf5'),
    'batch_size': 16,
    'epoch_cycles_train': 20,
    'num_epochs': 51,
    'evaluator': DummyEvaluator,
    'lr_decay': 1,
}
configuration = AttrDict(configuration)

model_config = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    n_ensemble_policies=1,
    n_processing_layers=3,
    n_classes=3,
    gripper_weights= [0.53, 0.07, 0.4], #butter - [0.7, 0.075, 0.225], #apple-[0.7, 0.05, 0.25],#full-[0.53, 0.07, 0.40],
    embed_mid_size=256,
    output_mid_size=256,
    lstm_hidden_size=256,
    skill_len=15,
    initial_bn=True,
    goal_conditioned=False,
    autoreg=True,
    gaussian_actions=False
)

# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec
data_config.dataset_spec.subseq_len = model_config.skill_len  # flat last action from seq gets cropped
data_config.dataset_spec.goal_len = 15
