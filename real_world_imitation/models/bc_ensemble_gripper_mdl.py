from contextlib import contextmanager
import itertools
import torch
import torch.nn as nn

from real_world_imitation.components.base_model import BaseModel
from real_world_imitation.modules.losses import NLL, L2Loss
from real_world_imitation.modules.subnetworks import Predictor
from real_world_imitation.utils.general_utils import AttrDict, ParamDict
from real_world_imitation.utils.pytorch_utils import no_batchnorm_update, no_dropout_update
from real_world_imitation.modules.variational_inference import ProbabilisticModel, MultivariateGaussian
from real_world_imitation.modules.layers import LayerBuilderParams

class BCEnsembleGripperMdl(BaseModel):
    """Ensemble of simple BC policies with Gaussian output distributions."""
    def __init__(self, params, logger=None):
        BaseModel.__init__(self, logger)
        ProbabilisticModel.__init__(self)
        self._hp = self._default_hparams()
        self._hp.overwrite(params)  # override defaults with config file
        self._hp.builder = LayerBuilderParams(self._hp.use_convs, self._hp.normalization, self._hp.dropout_rate)
        self.device = self._hp.device

        self.build_network()

    def _default_hparams(self):
        # put new parameters in here:
        return super()._default_hparams().overwrite(ParamDict({
            'use_convs': False,
            'dropout_rate': None,
            'device': None,
            'state_dim': -1,            # dimensionality of the state space
            'action_dim': -1,           # dimensionality of the action space
            'nz_mid': 128,              # number of dimensions for internal feature spaces
            'n_processing_layers': 5,   # number of layers in MLPs
            'n_ensemble_policies': 5,   # number of policies in ensemble
            'n_classes': 3, # number of classes to classify
            'gaussian_actions': True,
        }))

    def build_network(self):
        assert not self._hp.use_convs   # currently only supports non-image inputs

        action_dim = self._hp.action_dim*2 if self._hp.gaussian_actions == True else self._hp.action_dim
        self.ensemble = nn.ModuleList([
            Predictor(self._hp, input_size=self._hp.state_dim, output_size=action_dim + self._hp.n_classes)
            for _ in range(self._hp.n_ensemble_policies)])

    def forward(self, inputs):
        output = AttrDict()

        # run forward pass on each policy
        out_combined = torch.cat([policy(inputs.states[:,0].float()) for policy in self.ensemble], dim=0)

        output.pred_act = MultivariateGaussian(out_combined[:, :2*self._hp.action_dim]) if self._hp.gaussian_actions == True else out_combined[:, :self._hp.action_dim]
        if self._hp.n_classes>0:
            output.pred_discrete_act = out_combined[:, -self._hp.n_classes:]

        return output

    def loss(self, model_output, inputs):
        losses = AttrDict()

        # reconstruction loss
        if self._hp.gaussian_actions:
            losses.nll = NLL()(model_output.pred_act, torch.tile(inputs.actions[:, 0, :-1], (self._hp.n_ensemble_policies, 1)))
        else:
            losses.mse = L2Loss()(model_output.pred_act.float(), inputs.actions[:, 0, :-1].float())

        # cross entropy loss for gripper action classification
        if self._hp.n_classes>0:
            losses.cross_ent = AttrDict(value=nn.CrossEntropyLoss(weight=torch.tensor(self._hp.gripper_weights, device=self._hp.device))(model_output.pred_discrete_act, torch.tile(inputs.actions[:,0,-1].long(), (self._hp.n_ensemble_policies,))), weight=1)

        losses.total = self._compute_total_loss(losses)
        return losses

    def compute_actions(self, state):
        """Computes average variance of output distribution as well as within-ensemble KL div between outputs,"""
        # run all ensemble models over input
        assert len(state.shape) == 1
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, device=self._hp.device)
        with no_batchnorm_update(self.ensemble):
            with no_dropout_update(self.ensemble):
                with torch.no_grad():
                    out_combined = [policy(state[None].float())[0] for policy in self.ensemble]
        
        ensemble_outputs = [MultivariateGaussian(out[:2*self._hp.action_dim]) for out in out_combined] if self._hp.gaussian_actions else [out[:self._hp.action_dim] for out in out_combined]
        if self._hp.n_classes>0:
            discrete_outputs = [out[-self._hp.n_classes:] for out in out_combined]

        # compute average entropy
        avg_entropy = torch.tensor(0)
        if self._hp.gaussian_actions:
            avg_entropy = torch.stack([p.entropy() for p in ensemble_outputs]).sum(-1).mean() #remove sum(-1) when using entropy function from multivariate gaussian
        # avg_entropy = torch.stack([p.entropy() for p in ensemble_outputs]).mean(dim=0) #also changed the entropy funtion in variational_inference

        # compute average pairwise KL
        avg_divergence = torch.tensor(0)
        if self._hp.gaussian_actions:
            avg_divergence = torch.stack([p.kl_divergence(q).mean()
                                      for p, q in itertools.permutations(ensemble_outputs, r=2)]).mean() if self._hp.n_ensemble_policies > 1 else torch.Tensor([0])
        # avg_divergence = torch.stack([p.kl_divergence(q)
        #                               for p, q in itertools.permutations(ensemble_outputs, r=2)]).mean(dim=0) if self._hp.n_ensemble_policies > 1 else torch.Tensor([0])

        '''
        add gripper divergence and entropy here
        '''
        def kl_discrete(p, q):
            sm = nn.Softmax(dim=0)
            p,q = sm(p), sm(q)
            kl = p*torch.log(p/q)
            return kl.sum()
        
        #print(discrete_outputs)
        act = torch.argmax(discrete_outputs[0]).data.cpu().numpy() if self._hp.n_classes>0 else None
        # ent_gripper = torch.stack([-p*torch.log(p) for p in discrete_outputs]).mean()
        # kl_gripper = torch.stack([kl_discrete(p,q)
        #                               for p, q in itertools.permutations(discrete_outputs, r=2)]).mean() if self._hp.n_ensemble_policies > 1 else torch.Tensor([0])

        #ensemble_outputs[0].sample().data.cpu().numpy(),
        #ensemble_outputs[0].mu.data.cpu().numpy()
        return ensemble_outputs[0].mu.data.cpu().numpy() if self._hp.gaussian_actions else ensemble_outputs[0].data.cpu().numpy(), \
               act, \
               avg_divergence.data.cpu().numpy(), \
               avg_entropy.data.cpu().numpy()

    def reset_inference(unused_arg):
        pass

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
