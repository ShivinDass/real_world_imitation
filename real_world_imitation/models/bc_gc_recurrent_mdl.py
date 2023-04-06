from contextlib import contextmanager
import itertools
import torch
import torch.nn as nn

from real_world_imitation.components.base_model import BaseModel
from real_world_imitation.modules.losses import NLL, L2Loss
from real_world_imitation.utils.general_utils import AttrDict, ParamDict
from real_world_imitation.utils.pytorch_utils import no_batchnorm_update, no_dropout_update
from real_world_imitation.modules.variational_inference import ProbabilisticModel, MultivariateGaussian
from real_world_imitation.modules.layers import LayerBuilderParams
from real_world_imitation.components.checkpointer import CheckpointHandler, freeze_module
from real_world_imitation.modules.recurrent_policy import RecurrentPolicy

class BCGoalConditionedRecurrentMdl(BaseModel):
    """Ensemble of simple BC policies with Gaussian output distributions."""
    def __init__(self, params, logger=None):
        BaseModel.__init__(self, logger)
        ProbabilisticModel.__init__(self)
        self._hp = self._default_hparams()
        self._hp.overwrite(params)  # override defaults with config file
        self._hp.builder = LayerBuilderParams(self._hp.use_convs, self._hp.normalization, self._hp.dropout_rate)
        self.device = self._hp.device

        self.build_network()
        self.reset_inference()

    def _default_hparams(self):
        # put new parameters in here:
        return super()._default_hparams().overwrite(ParamDict({
            'use_convs': False,
            'dropout_rate': None,
            'device': None,
            'state_dim': -1,            # dimensionality of the state space
            'goal_dim': -1,             # dimensionality of the goal space
            'action_dim': -1,           # dimensionality of the action space
            'gaussian_actions': False,
            'n_processing_layers': 5,   # number of layers in MLPs
            'n_ensemble_policies': 5,   # number of policies in ensemble
            'n_classes': 3,             # number of classes to classify for discrete actions
            'gripper_weights': [0.585, 0.08, 0.335],
            'embed_mid_size': 256,      # mid size for embed MLP
            'lstm_hidden_size': 256,          # hidden dim size for LSTM
            'output_mid_size': 256,
            'skill_len': -1,            # skill horizon
            'initial_bn': True,
            'goal_conditioned': True,
            'autoreg': True,
        }))

    def build_network(self):
        assert not self._hp.use_convs   # currently only supports non-image inputs

        self.goal_generator = None
        self.ensemble = nn.ModuleList([RecurrentPolicy(self._hp) for _ in range(self._hp.n_ensemble_policies)])

    def forward(self, inputs):
        '''
        input: AttrDict containing states(BatchxTimexDim), actions(BatchxTimexDim) and goals(BatchxTimexDim)
        '''

        output = AttrDict()

        goals = inputs.goals[:, :, -self._hp.goal_dim:]

        # generate recurrent actions
        actions = torch.cat([policy(inputs.states, goals) for policy in self.ensemble], dim=0)

        output.pred_act = MultivariateGaussian(actions[:, :2*self._hp.action_dim]) if self._hp.gaussian_actions == True else actions[:, :self._hp.action_dim]
        if self._hp.n_classes > 0:
            output.pred_discrete_act = actions[:, -self._hp.n_classes:]
        
        return output

    def loss(self, model_output, inputs):
        losses = AttrDict()

        # reconstruction loss
        target_actions = inputs.actions.repeat(self._hp.n_ensemble_policies, 1, 1)
        if self._hp.gaussian_actions == True:
            losses.nll = NLL()(model_output.pred_act, target_actions[:, :, :-1].reshape(-1, self._hp.action_dim))
        else:
            losses.mse = L2Loss()(model_output.pred_act, target_actions[:, :, :-1].reshape(-1, self._hp.action_dim))
        
        # cross entropy loss for gripper action classification
        if self._hp.n_classes > 0:
            discrete_labels = target_actions[:,:,-1].reshape(-1,).long()
            losses.cross_ent = AttrDict(
                        value=nn.CrossEntropyLoss(weight=torch.tensor(self._hp.gripper_weights, dtype=torch.float32, device=self.device))(model_output.pred_discrete_act, discrete_labels),
                        weight=1
                    )

        losses.total = self._compute_total_loss(losses)
        return losses
    
    def reset(self):
        self.reset_inference()
        
    def reset_inference(self):
        self.cur_step = 0
        self.h0 = [None for _ in range(self._hp.n_ensemble_policies)]
        self.c0 = [None for _ in range(self._hp.n_ensemble_policies)]

    def compute_actions(self, state):
        assert len(state.shape) == 1
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, device=self.device)
        
        if self.cur_step >= self._hp.skill_len:
            self.reset_inference()

        with no_batchnorm_update(nn.ModuleList([self.ensemble])):
            with no_dropout_update(nn.ModuleList([self.ensemble])):
                with torch.no_grad():
                    if self.cur_step==0:
                        self.local_state = state.clone()
                        self.local_goal = self.get_goal(state[None])

                    ensemble_outputs = []
                    discrete_output= []
                    for i, policy in enumerate(self.ensemble):
                        if self._hp.autoreg:
                            actions, self.h0[i], self.c0[i] = policy.compute_action(self.local_state[None].clone(), self.local_goal.clone(), self.h0[i], self.c0[i])
                        else:
                            actions, self.h0[i], self.c0[i] = policy.compute_action(state[None], self.local_goal.clone(), self.h0[i], self.c0[i])

                        ensemble_outputs.append(MultivariateGaussian(actions[0, :2*self._hp.action_dim]) if self._hp.gaussian_actions else actions[0, :self._hp.action_dim])
                        if self._hp.n_classes > 0:
                            discrete_output.append(actions[0, -self._hp.n_classes:])
                    
        # outputs are the mean of all ensemble actions
        output = torch.stack([out.mu for out in ensemble_outputs])[0] if self._hp.gaussian_actions else torch.stack(ensemble_outputs).mean(dim=0)
        discrete_output = torch.argmax(torch.stack(discrete_output).mean(dim=0)).data.cpu().numpy() if self._hp.n_classes > 0 else None
        
        # compute future state entropy
        avg_entropy = self.goal_generator.get_entropy(state[None])

        # compute average divergence
        if self._hp.gaussian_actions:
            avg_divergence = torch.stack([p.kl_divergence(q).mean()
                            for p, q in itertools.permutations(ensemble_outputs, r=2)]).mean() if self._hp.n_ensemble_policies > 1 else torch.Tensor([0])
        else:
            avg_divergence = torch.stack(ensemble_outputs).std(dim=0).mean()
        
        self.cur_step += 1

        return  output.data.cpu().numpy(), \
                discrete_output, \
                avg_divergence.data.cpu().numpy(), \
                avg_entropy.data.cpu().numpy()

    def get_goal(self, state):
        assert self.goal_generator is not None
        return self.goal_generator.generate_goal(state)

    def info(self):
        """Optionally return any info that should be stored in the episode loop."""
        return AttrDict()

    @property
    def resolution(self):
        return 64  # return dummy resolution, images are not used by this model

    @contextmanager
    def val_mode(self):
        pass
        yield
        pass
    
    def log_outputs(self, model_output, inputs, losses, step, log_images, phase, **logging_kwargs):
        super().log_outputs(model_output, inputs, losses, step, log_images, phase, **logging_kwargs)

    def load_goal_gen_and_freeze(self, model_class, config, ckpt_path, epoch):
        assert self._hp.goal_dim == config.state_dim
        
        self.goal_generator = model_class(config)
        CheckpointHandler.load_weights(CheckpointHandler.get_resume_ckpt_file(epoch, ckpt_path), self.goal_generator)
        freeze_module(self.goal_generator)
    
    def get_lstm_hidden_units(self, states, goals):
        with no_batchnorm_update(nn.ModuleList([self.ensemble])):
            with no_dropout_update(nn.ModuleList([self.ensemble])):
                with torch.no_grad():
                    _, h, c = self.ensemble[0].compute_action(states, goals)
        return h
