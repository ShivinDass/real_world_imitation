import os

from real_world_imitation.models.bc_ensemble_mdl import BCEnsembleMdl
from real_world_imitation.models.bc_ensemble_gripper_mdl import BCEnsembleGripperMdl
from real_world_imitation.models.bc_deterministic_mdl import DeterministicBCMdl
from real_world_imitation.models.bc_gmm import BCEnsembleGMM
from real_world_imitation.models.iris import IRIS
from real_world_imitation.components.logger import Logger
from real_world_imitation.utils.general_utils import AttrDict
from real_world_imitation.configs.default_data_configs.real_world import data_spec
from real_world_imitation.components.evaluator import DummyEvaluator


current_dir = os.path.dirname(os.path.realpath(__file__))


configuration = {
    'model': BCEnsembleGripperMdl,
    'logger': Logger,
    'data_dir': os.path.join(os.environ['DATA_DIR'], 'task1_task2_embedded50_clean_gripper/embedded50_clean_gripper.hdf5'),
    'batch_size': 64,
    'epoch_cycles_train': 20,
    'num_epochs': 50,
    'evaluator': DummyEvaluator,
    'lr_decay': 1,
}
configuration = AttrDict(configuration)

model_config = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    # dropout_rate=0.1,
    nz_mid=256,
    n_ensemble_policies=1,
    n_processing_layers=5,
    n_gaussians=3,
    n_classes=3,
    gaussian_actions=False,
    gripper_weights=[0.585, 0.08, 0.335],
)

configuration['batch_size'] *= model_config.n_ensemble_policies     # since batch will be chunked per policy

# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec
data_config.dataset_spec.subseq_len = 1  # flat last action from seq gets cropped
