import os

from real_world_imitation.models.bc_ensemble_gripper_mdl import BCEnsembleGripperMdl
from real_world_imitation.components.logger import Logger
from real_world_imitation.utils.general_utils import AttrDict
from real_world_imitation.configs.default_data_configs.real_world import data_spec
from real_world_imitation.components.evaluator import DummyEvaluator


current_dir = os.path.dirname(os.path.realpath(__file__))


configuration = {
    'model': BCEnsembleGripperMdl,
    'logger': Logger,
    'data_dir': os.path.join(os.environ['DATA_DIR'], 'boosting_seed_embedded50_clean_gripper/embedded50_clean_gripper.hdf5'),
    'batch_size': 64,
    'epoch_cycles_train': 20,
    'num_epochs': 10,
    'evaluator': DummyEvaluator,
    'lr_decay': 1,
}
configuration = AttrDict(configuration)

model_config = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    # dropout_rate=0.1,
    nz_mid=256,
    n_ensemble_policies=2,
    n_processing_layers=5,
    n_classes=3,
    gripper_weights=[0.585, 0.08, 0.335],
    gaussian_actions=True,
)

# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec
data_config.dataset_spec.subseq_len = 1  # flat last action from seq gets cropped
data_config.dataset_spec.goal_len = 1