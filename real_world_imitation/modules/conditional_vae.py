from real_world_imitation.data.furniture.src.furniture_data_loader import OculusVRSequenceSplitDataset
import torch
import torch.nn as nn
import numpy as np

from real_world_imitation.components.base_model import BaseModel
from real_world_imitation.utils.pytorch_utils import no_batchnorm_update, no_dropout_update, RAdam, TensorModule
from real_world_imitation.modules.layers import LayerBuilderParams
from real_world_imitation.modules.subnetworks import Predictor
from real_world_imitation.modules.variational_inference import MultivariateGaussian
from real_world_imitation.modules.mdn import MDN, GMM
from real_world_imitation.utils.general_utils import AttrDict, ParamDict, get_clipped_optimizer

class cVAE(BaseModel):

    def __init__(self, params, logger=None):
        BaseModel.__init__(self, logger)
        self._hp = self._default_hparams()
        self._hp.overwrite(params)
        self.device = self._hp.device

        self._hp.builder = LayerBuilderParams(self._hp.use_convs, normalization='none')
        self.num_gaussians = 2
        
        if self._hp.target_kl is None:
            self._log_beta = TensorModule(np.log(self._hp.kl_div_weight)
                                          * torch.ones(1, requires_grad=False, device=self._hp.device))
            self._beta_opt = None
        else:
            self._log_beta = TensorModule(torch.log(self._hp.kl_div_weight*torch.ones(1, requires_grad=True, device=self._hp.device)))
            self._beta_opt = self._get_beta_opt()

        self.build_network()

    def _default_hparams(self):
        # put new parameters in here:
        default_dict =  ParamDict({
            'use_convs': False,
            'dropout_rate': None,
            'device': None,
            'state_dim': -1,            # dimensionality of the state space
            'action_dim': -1,           # dimensionality of the action space
            'condition_dim': -1,        # dimensionality of the condition
            'vae_mid': 128,              # number of dimensions for internal feature spaces
            'vae_latent_dim': 32,
            'vae_processing_layers': 5,   # number of layers in MLPs
            'vae_learnt_prior': None,   # default is fixed prior multivariate gaussian
            'vae_prior_layers': 2,
            'kl_div_weight': 5e-3,
            'target_kl': None,           # Target KL for sigma-vae
            'n_sample_goals': 1024
        })

        return default_dict



    def build_network(self):
        assert not self._hp.use_convs

        if self._hp.vae_learnt_prior == MultivariateGaussian:
            self.prior = Predictor(self._hp, input_size=self._hp.condition_dim, output_size=self._hp.vae_latent_dim*2, mid_size=self._hp.vae_mid, num_layers=self._hp.vae_prior_layers)
        elif self._hp.vae_learnt_prior == GMM:
            self.prior = nn.Sequential(
                Predictor(self._hp, input_size=self._hp.condition_dim, output_size=self._hp.vae_mid, mid_size=self._hp.vae_mid, num_layers=self._hp.vae_prior_layers),
                MDN(input_size=self._hp.vae_mid, output_size=self._hp.vae_latent_dim, num_gaussians=self.num_gaussians)
            )
        else:
            self.prior = torch.cat((torch.zeros(self._hp.vae_latent_dim, device=self._hp.device), torch.ones(self._hp.vae_latent_dim, device=self._hp.device)))

        self.encoder = Predictor(self._hp, input_size=self._hp.state_dim + self._hp.condition_dim, output_size=self._hp.vae_latent_dim*2, mid_size=self._hp.vae_mid, num_layers=self._hp.vae_processing_layers)
        self.decoder = Predictor(self._hp, input_size=self._hp.vae_latent_dim + self._hp.condition_dim, output_size=self._hp.state_dim, mid_size=self._hp.vae_mid, num_layers=self._hp.vae_processing_layers)

    def forward(self, inputs):
        output = AttrDict()

        states = inputs.states.view(-1, self._hp.condition_dim)
        goals = inputs.goals.view(-1, self._hp.state_dim)

        output.latent = MultivariateGaussian(self.encoder(torch.cat((goals, states.float()), dim=1)))
        
        sampled_latent = output.latent.sample()
        output.pred_goal = self.decoder(torch.cat((sampled_latent, states), dim=1))

        return output

    def loss(self, model_output, inputs):
        losses = AttrDict()
        losses.recon_loss = AttrDict(
                    value=nn.MSELoss()(model_output.pred_goal, inputs.goals.view(-1, self._hp.state_dim)),
                    weight=1
                )

        if not self._hp.vae_learnt_prior == GMM:
            losses.kl_loss = AttrDict(
                        value=model_output.latent.kl_divergence(self.get_prior(inputs.states.view(-1, self._hp.condition_dim))).mean(),
                        weight=self._log_beta().exp()[0].detach()
                    )
            losses.total = self._compute_total_loss(losses)
        
        # kl approximation from sampling
        z = model_output.latent.sample()
        sample_kl_loss = (model_output.latent.log_prob(z) - self.get_prior(inputs.states.view(-1, self._hp.condition_dim)).log_prob(z))

        losses.kl_sample_loss = AttrDict(
                value=sample_kl_loss.mean(),
                weight=self._log_beta().exp()[0].detach()*1e-2
            )
        if self._hp.vae_learnt_prior == GMM:
            losses.total = self._compute_total_loss(losses)
            losses.kl_loss = AttrDict(value=torch.tensor(0), weight=0)
        
        if self._hp.target_kl is not None:
            if self._hp.vae_learnt_prior == GMM:
                self._update_beta(losses.kl_sample_loss.value)
            else:
                self._update_beta(losses.kl_loss.value)

        return losses

    def get_prior(self, state):
        if self._hp.vae_learnt_prior == None:
                return MultivariateGaussian(self.prior.repeat(state.shape[0],1))
        with no_batchnorm_update(self.prior):
            with no_dropout_update(self.prior):
                with torch.no_grad():
                    return self._hp.vae_learnt_prior(self.prior(state))


    def sample_latent_from_prior(self, state):
        prior =  self.get_prior(state)
        return prior.sample()

    def generate_goal(self, state):
        with no_batchnorm_update(self.decoder):
            with no_dropout_update(self.decoder):
                with torch.no_grad():
                    sampled_latent = self.sample_latent_from_prior(state)
                    goal = self.decoder(torch.cat((sampled_latent, state), dim=1))
        return goal

    def _get_beta_opt(self):
        return get_clipped_optimizer(filter(lambda p: p.requires_grad, self._log_beta.parameters()),
                                     lr=self._hp.beta_update_weight, optimizer_type=RAdam, betas=(0.9, 0.999), gradient_clip=None)

    def _update_beta(self, kl_div):
        """Updates beta with dual gradient descent."""
        if self._hp.target_kl is not None:
            beta_loss = self._log_beta().exp() * (self._hp.target_kl - kl_div).detach().mean()
            self._beta_opt.zero_grad()
            beta_loss.backward()
            self._beta_opt.step()
    
    def get_entropy(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, device=self.device)
        EEF_IDXS = [-13, -12, -11] #[-15, -14, -13]
        goals = self.generate_goal(state.repeat(self._hp.n_sample_goals, 1))

        return torch.std(goals[:, EEF_IDXS], dim=0).mean()

if __name__=='__main__':
    WANDB_PROJECT_NAME = 'assisted_teleop'
    WANDB_ENTITY_NAME = 'clvr'

    EXP_PATH = 'assisted_teleop/vae_model/blocks/newData_fixgripper_L2_G7_tKL0005'

    EEF_IDXS = [-15, -14, -13]#[-13, -12, -11] # [-15, -14, -13]
    PLOT_LIMS = [(-0.35, 0.25), (-0.7, -0.25), (0.15, 0.45)]
    # PLOT_LIMS = [(-0.25, 0.25), (-0.2, 0.2), (0., 0.3)]


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_printoptions(precision=6)

    import os, cv2, h5py
    import matplotlib.pyplot as plt
    from real_world_imitation.configs.default_data_configs.real_world import data_spec
    # from assisted_teleop.configs.default_data_configs.furniture import data_spec
    from real_world_imitation.utils.general_utils import map_dict, dictlist2listdict
    from real_world_imitation.utils.vis_utils import add_captions_to_seq
    from real_world_imitation.components.checkpointer import CheckpointHandler, save_cmd, save_git
    from real_world_imitation.utils.wandb import WandBLogger
    from real_world_imitation.train import save_checkpoint
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    import matplotlib
    # matplotlib.use('TkAgg')

    def display_multiple_img(batch):
        windows = int(np.sqrt(batch.shape[0]))

        plt.figure(figsize=(10,10))
        for i in range(windows*windows):
            ax = plt.subplot(windows,windows,i+1)
            plt.imshow(cv2.cvtColor(np.float32(batch[i].movedim(0,2).cpu().numpy()), cv2.COLOR_BGR2RGB))
            plt.axis("off")
        plt.show()
    
    def calculate_variance(batch):
        return torch.std(batch[:, EEF_IDXS], dim=0).mean().data.cpu().numpy()
    
    def get_eef_plot(state_seq, pred_goals):
        X, Y, Z = pred_goals.data.cpu().numpy().transpose(1,0)
        sx, sy, sz = (state_seq*val_dataset.input_stddev[-15:] + val_dataset.input_mean[-15:])[:, EEF_IDXS].transpose(1,0)
        fig = plt.figure('eefpos_projection')
        plt.clf()
        canvas = FigureCanvas(fig)
        ax1 = plt.subplot(121, projection='3d')
        ax1.set_xlim3d(left=PLOT_LIMS[0][0], right=PLOT_LIMS[0][1])
        ax1.set_ylim3d(bottom=PLOT_LIMS[1][0], top=PLOT_LIMS[1][1])
        ax1.set_zlim3d(bottom=PLOT_LIMS[2][0], top=PLOT_LIMS[2][1])
        ax1.scatter(X, Y, Z, marker='o', color=(0.5, 0.5, 1.), alpha=0.5)
        ax1.scatter(sx, sy, sz, marker='o', color='red')

        ax2 = plt.subplot(122, projection='3d')
        ax2.set_xlim3d(left=PLOT_LIMS[0][0], right=PLOT_LIMS[0][1])
        ax2.set_ylim3d(bottom=PLOT_LIMS[1][0], top=PLOT_LIMS[1][1])
        ax2.set_zlim3d(bottom=PLOT_LIMS[2][0], top=PLOT_LIMS[2][1])
        ax2.scatter(X, Y, Z, marker='o', color=(0.5, 0.5, 1.), alpha=0.5)
        ax2.scatter(sx, sy, sz, marker='o', color='red')
        ax2.view_init(elev=10, azim=0)
        
        canvas.draw()
        width, height = fig.get_size_inches()*fig.get_dpi()
        return np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)


    def setup_logging(conf, exp_path):
        print('Writing to the experiment directory: {}'.format(exp_path))
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
        save_cmd(exp_path)
        save_git(exp_path)

        exp_name = f"{'_'.join(conf.exp_path.split('/'))}"
        writer = WandBLogger(exp_name, WANDB_PROJECT_NAME, entity=WANDB_ENTITY_NAME,
                                path=exp_path, conf=conf, exclude=['model_rewards', 'data_dataset_spec_rewards'])
        return writer

    def log_inference(logger, metrics, epoch):
        log_prefix = 'inference/epoch{}'.format(epoch)
        
        fig = plt.figure()
        plt.plot(metrics.eef_variance)
        logger.log_plot(fig, f"{log_prefix}/eef_variance")
        plt.close(fig)

        info = dictlist2listdict(AttrDict(eef_var=metrics.eef_variance))
        
        vid = np.stack(add_captions_to_seq(metrics.eef_projections, info))
        logger.log_videos([vid.transpose(0, 3, 1, 2)], name=f"{log_prefix}/eef_projection")
        
        if len(metrics.img_list) > 0:
            vid = np.stack(add_captions_to_seq(metrics.img_list, info))
            logger.log_videos([vid.transpose(0, 3, 1, 2)], name=f"{log_prefix}/rollouts")    
        if len(metrics.img_pred_goal_list) > 0:
            logger.log_videos([np.stack(metrics.img_pred_goal_list).transpose(0, 3, 1, 2)], name=f"{log_prefix}/predicted_goals")

    # TRAIN CONFIGS
    conf = AttrDict({
        'exp_path': 'vae_model/real_world/boosting_boosted_kl_target_0.01_proprioceptivegoal_fullstate_subseqlen15',
        'batch_size': 16,
        'data_dir': os.path.join(os.environ['DATA_DIR'], 'boosting_boosted_embedded50_clean_gripper/embedded50_clean_gripper.hdf5'),
        'num_epochs': 251
    })
    exp_path = os.path.join(os.environ['EXP_DIR'], conf.exp_path)

    # LOAD DATA
    data_conf = AttrDict()
    data_conf.dataset_spec = data_spec
    data_conf.dataset_spec.subseq_len = 15
    data_conf.dataset_spec.goal_len = 15
    data_conf.device = device
    dataset = data_spec.dataset_class(data_dir = conf.data_dir,
                                data_conf= data_conf, phase='train')
    val_dataset = data_spec.dataset_class(data_dir = conf.data_dir,
                                data_conf= data_conf, phase='val')
    loader = torch.utils.data.DataLoader(dataset, batch_size=conf.batch_size, shuffle=True)

    # CREATE cVAE
    vae_config = AttrDict(
        state_dim=15,#
        action_dim=data_spec.n_actions,
        condition_dim=data_spec.state_dim,
        device=device,
        vae_latent_dim=128,
        vae_processing_layers=5,
        vae_learnt_prior=None,
        vae_prior_layers=5, # number of layers if prior is learnt
        vae_mid=128,
        kl_div_weight=5e-3,
        target_kl = 0.01,
        beta_update_weight=1e-3,
        n_sample_goals=1024
    )
    logger = setup_logging(conf, exp_path)
    model = cVAE(params=vae_config, logger=logger).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    global_step = 0
    # TRAINING
    for epoch in range(conf.num_epochs):
        total_loss=0
        kl_loss = 0
        kl_sample_loss = 0
        for i, sampled_batch in enumerate(loader):
            inputs = AttrDict(map_dict(lambda x: x.to(device), sampled_batch))
            
            # inputs.states = inputs.states[:, :, :-8]#
            if vae_config.state_dim==15:
                inputs.goals = inputs.goals[:, :, -15:]
            
            optimizer.zero_grad()
            output = model(inputs)
            losses = model.loss(output, inputs)
            losses.total.value.backward()
            optimizer.step()

            total_loss += losses.total.value.detach()/len(loader)
            kl_loss += losses.kl_loss.value.detach()/len(loader)
            kl_sample_loss += losses.kl_sample_loss.value.detach()/len(loader)

            global_step += 1

        # log outputs
        print("epoch{} loss:".format(epoch), total_loss, kl_loss, kl_sample_loss)
        print("beta:", model._log_beta().exp().data)
        if logger is not None:
            model.log_outputs(output, inputs, losses, global_step, log_images=False, phase='train')
        
        # VALIDATION
        if epoch%50==0 and epoch>0:
            
            if logger is not None:
                save_checkpoint({
                    'epoch': epoch,
                    'global_step': global_step,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                },  os.path.join(exp_path, 'weights'), CheckpointHandler.get_ckpt_name(epoch))
            
            sample_goals = 1024
            with torch.no_grad():
                for _ in range(3):
                    logging_metrics = AttrDict(
                        img_list = [],
                        eef_variance = [],
                        eef_projections = [],
                        img_pred_goal_list = []
                    )
                    
                    seq = val_dataset._sample_seq()
                    for i, obs in enumerate(seq.states):
                        states = torch.tensor(obs).to(device).reshape(1, -1).repeat(sample_goals, 1).float()
                        goal = torch.tensor(seq.states[min(len(seq.states)-1, i+data_conf.dataset_spec.subseq_len)]).to(device)[None, :].float()

                        pred_goals = model.generate_goal(states)#
                        pred_goals[:, -15:] = pred_goals[:, -15:]*torch.tensor(val_dataset.input_stddev[-15:], device=device) + torch.tensor(val_dataset.input_mean[-15:], device=device)
                        
                        logging_metrics.eef_variance.append(calculate_variance(pred_goals))
                        logging_metrics.eef_projections.append(get_eef_plot(seq.states[:i+1, -15:], pred_goals[:, EEF_IDXS]))

                    if logger is not None:
                        log_inference(logger, logging_metrics, epoch=epoch)                
                continue
