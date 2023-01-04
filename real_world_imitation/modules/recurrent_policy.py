import torch
import torch.nn as nn

from real_world_imitation.modules.subnetworks import Predictor
from real_world_imitation.modules.layers import LayerBuilderParams

class RecurrentPolicy(nn.Module):
    def __init__(self, params):
        super().__init__()
        self._hp = params

        self._hp.builder = LayerBuilderParams(self._hp.use_convs, self._hp.normalization, self._hp.dropout_rate)
        self.build_network()

    def build_network(self):
        if self._hp.goal_conditioned:
            self.embed_net_state = Predictor(self._hp, input_size = self._hp.state_dim, output_size = self._hp.embed_mid_size, mid_size=self._hp.embed_mid_size, num_layers=2, initial_batchnorm=self._hp.initial_bn)
            self.embed_net_goal = Predictor(self._hp, input_size = self._hp.goal_dim, output_size = self._hp.embed_mid_size, mid_size=self._hp.embed_mid_size, num_layers=2, initial_batchnorm=self._hp.initial_bn)
            self.embed_net = Predictor(self._hp, input_size = self._hp.embed_mid_size*2, output_size = self._hp.lstm_hidden_size, mid_size=self._hp.embed_mid_size, num_layers=2, final_activation=nn.LeakyReLU(0.2, inplace=True), initial_batchnorm=self._hp.initial_bn)
        else:
            self.embed_net = Predictor(self._hp, input_size = self._hp.state_dim, output_size = self._hp.lstm_hidden_size, mid_size=self._hp.embed_mid_size, num_layers=2, final_activation=nn.LeakyReLU(0.2, inplace=True), initial_batchnorm=self._hp.initial_bn)
        
        self.lstm = nn.LSTM(input_size = self._hp.lstm_hidden_size, hidden_size = self._hp.lstm_hidden_size, batch_first=True)
        action_dim = self._hp.action_dim*2 if self._hp.gaussian_actions == True else self._hp.action_dim
        self.output_net = Predictor(self._hp, input_size = self._hp.lstm_hidden_size, output_size = action_dim + self._hp.n_classes, mid_size=self._hp.output_mid_size, initial_batchnorm=self._hp.initial_bn)

    def forward(self, states, goals):
        batch_size, seq_len = states.shape[:2]

        # concatenate the current and goal state for each sequence to create goal_conditioned state
        if self._hp.autoreg:
            states = torch.repeat_interleave(states[:, 0, :][:, None], repeats=seq_len, dim=1)
        
        if self._hp.goal_conditioned:
            goal_states = torch.repeat_interleave(goals[:, 0, :][:, None], repeats=seq_len, dim=1)

            st = self.embed_net_state(states.view(-1, self._hp.state_dim))
            gl = self.embed_net_goal(goal_states.view(-1, self._hp.goal_dim))
            embed = torch.cat((st, gl), dim=-1)
        else:
            embed = states.view(-1, self._hp.state_dim)

        embed = self.embed_net(embed)
        
        # run embedded states through lstm
        lstm_out, _ = self.lstm(embed.view(batch_size, seq_len, -1))
        
        # generate actions
        actions = self.output_net(lstm_out.reshape(batch_size*seq_len, -1))

        return actions

    def compute_action(self, state, goal, h0=None, c0=None):
        
        if self._hp.goal_conditioned:
            embed = self.embed_net(torch.cat((self.embed_net_state(state), self.embed_net_goal(goal)), dim=-1)).view(1,1,-1)
        else:
            embed = self.embed_net(state).view(1,1,-1)
        
        lstm_out, (h0, c0) = self.lstm(embed, (h0, c0)) if h0 is not None else self.lstm(embed)
        action = self.output_net(lstm_out.reshape(1,-1))

        return action, h0, c0