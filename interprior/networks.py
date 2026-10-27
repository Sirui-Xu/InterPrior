# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import layers
from rl_games.algos_torch import network_builder
import torch
import torch.nn as nn
import numpy as np

DISC_LOGIT_INIT_SCALE = 1.0


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).permute(0, 2, 1)
        self.pe = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        x = x + self.pe[:, : x.shape[1], : x.shape[2]]
        return x


class InterPriorNetworkBuilder(network_builder.A2CBuilder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    class Network(network_builder.A2CBuilder.Network):

        def __init__(self, params, **kwargs):
            super().__init__(params, **kwargs)
            if self.is_continuous:
                if not self.space_config["learn_sigma"]:
                    actions_num = kwargs.get("actions_num")
                    sigma_init = self.init_factory.create(
                        **self.space_config["sigma_init"]
                    )
                    self.sigma = nn.Parameter(
                        torch.zeros(
                            actions_num, requires_grad=False, dtype=torch.float32
                        ),
                        requires_grad=False,
                    )
                    sigma_init(self.sigma)
            input_shape_goal = 628
            input_shape_obs = 1127
            self.vae_dim = self.space_config["vae_dim"]
            self.freeze = self.space_config["freeze"]
            self.input_shape_goal = 628
            self.input_shape_obs = 1127
            ff_size = 1024
            num_channels = 512
            num_heads = 4
            self.MLPEmbedding_goal = nn.Linear(input_shape_goal, num_channels)
            self.MLPEmbedding_obs = nn.Linear(input_shape_obs, num_channels)
            self.PositionalEmbedding = PositionalEncoding(d_model=num_channels)
            from torch.nn import TransformerEncoderLayer

            seqTransEncoderLayer = TransformerEncoderLayer(
                d_model=num_channels,
                nhead=num_heads,
                dim_feedforward=ff_size,
                dropout=0,
                activation="relu",
                batch_first=False,
            )
            self.prior = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=4)
            self._mu_head_1 = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, self.vae_dim),
            )
            self._logvar_head_1 = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, self.vae_dim),
            )
            self._mu_head_2 = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, self.vae_dim),
            )
            self._logvar_head_2 = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, self.vae_dim),
            )
            self.encoder = nn.Sequential(
                nn.Linear(input_shape_obs + 2 * 4 * input_shape_goal, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
            )
            self.actor_encoder = nn.Sequential(
                nn.Linear(
                    input_shape_obs * 5 + self.vae_dim,
                    input_shape_goal + input_shape_obs,
                ),
                nn.ReLU(),
            )
            self.pred_decoder = nn.Sequential(
                nn.Linear(input_shape_goal + input_shape_obs, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, input_shape_goal),
            )
            self.cls_emb = nn.Parameter(
                torch.randn(1, 1, num_channels), requires_grad=True
            )
            if self.freeze:
                self.freeze_trunk()
            return

        def forward(self, obs_dict):
            if obs_dict["with_vae"]:
                actor_outputs, states = self.get_action_and_vae_outputs(obs_dict)
            else:
                actor_outputs = self.act(obs_dict)
                states = obs_dict.get("rnn_states", None)
            obs = obs_dict["obs"]
            value = self.eval_critic(obs_dict)
            output = actor_outputs + (value, states)
            return output

        def _prior(self, input_dict):
            obs = input_dict["obs"]
            mask = input_dict["mask"]
            obs_mask = obs[..., obs.shape[-1] - mask.shape[-1] :] * mask
            obs_mask = obs_mask.view(obs_mask.shape[0], 4, -1)
            a_in = torch.cat(
                [
                    self.cls_emb.expand(obs_mask.shape[0], 1, -1),
                    self.MLPEmbedding_obs(
                        obs[..., : obs.shape[-1] - mask.shape[-1]]
                        .unsqueeze(1)
                        .view(obs_mask.shape[0], 5, -1)
                    ),
                    self.MLPEmbedding_goal(obs_mask),
                ],
                dim=1,
            )
            a_out = self.PositionalEmbedding(a_in)
            a_out = a_out.permute(1, 0, 2).contiguous()
            a_out = self.prior(a_out)[0]
            mu = self._mu_head_1(a_out)
            logvar = self._logvar_head_1(a_out)
            return {"mu": mu, "logvar": logvar}

        def _encoder(self, input_dict):
            obs = input_dict["obs"]
            mask = input_dict["mask"]
            obs_mask = obs[..., obs.shape[-1] - mask.shape[-1] :] * mask
            obs_view = torch.cat(
                [obs[..., 4 * self.input_shape_obs :], obs_mask], dim=-1
            )
            a_out = self.encoder(obs_view)
            mu = self._mu_head_2(a_out)
            logvar = self._logvar_head_2(a_out)
            return {"mu": mu, "logvar": logvar}

        def freeze_trunk(self):
            to_freeze = [self.actor_encoder, self.pred_decoder, self.actor_mlp, self.mu]
            for m in to_freeze:
                if hasattr(m, "parameters"):
                    for p in m.parameters():
                        p.requires_grad = False
            if isinstance(getattr(self, "sigma", None), torch.nn.Parameter):
                self.sigma.requires_grad = False
            for m in to_freeze:
                if hasattr(m, "eval"):
                    m.eval()

        def _trunk(self, input_dict):
            a_out = torch.cat(
                [
                    input_dict["vae_latent"],
                    input_dict["obs"][
                        ...,
                        : input_dict["obs"].shape[-1] - input_dict["mask"].shape[-1],
                    ],
                ],
                dim=-1,
            )
            a_out = self.actor_encoder(a_out)
            p_out = self.pred_decoder(a_out)
            a_out = self.actor_mlp(a_out)
            mu = self.mu_act(self.mu(a_out))
            if torch.any(~torch.isfinite(mu)):
                raise Exception("invalid mu")
                mu = torch.where(~torch.isfinite(mu), torch.zeros_like(mu), mu)
            if self.space_config["fixed_sigma"]:
                sigma = mu * 0.0 + self.sigma_act(self.sigma)
            else:
                sigma = self.sigma_act(self.sigma(a_out))
            if torch.any(~torch.isfinite(sigma)):
                raise Exception("invalid sigma")
                sigma = torch.where(
                    ~torch.isfinite(sigma), torch.zeros_like(sigma), sigma
                )
            return ((mu, sigma), p_out)

        def eval_critic(self, input_dict):
            obs = input_dict["obs"]
            mask = input_dict["mask"]
            obs_mask = obs[..., obs.shape[-1] - mask.shape[-1] :] * mask
            obs_mask = obs_mask.view(obs_mask.shape[0], 4, -1)
            c_out = self.critic_cnn(
                torch.cat([obs[..., : self.input_shape_obs], obs_mask[:, -1]], dim=1)
            )
            c_out = c_out.contiguous().view(c_out.size(0), -1)
            c_out = self.critic_mlp(c_out)
            value = self.value_act(self.value(c_out))
            return value

        def reparameterization(self, mean, std, vae_noise):
            z = mean + std * vae_noise
            norm_detached = z.norm(dim=-1, keepdim=True).detach() + 1e-08
            z = z / norm_detached
            return z

        def act(self, input_dict: dict):
            prior_out = self._prior(input_dict)
            if input_dict["with_encoder"]:
                encoder_out = self._encoder(input_dict)
                mu = prior_out["mu"] + encoder_out["mu"]
                logvar = encoder_out["logvar"]
            else:
                mu = prior_out["mu"]
                logvar = prior_out["logvar"]
            z = self.reparameterization(
                mu, torch.exp(0.5 * logvar), input_dict["vae_noise"]
            )
            input_dict["vae_latent"] = z
            action, prediction = self._trunk(input_dict)
            return action

        def get_action_and_vae_outputs(self, input_dict: dict):
            """Get action and VAE outputs by sampling from the encoder.

            This method is used during training to sample from both the encoder and prior networks.
            The encoder output's mu acts as a residual to the prior's mu, while its logvar is used directly.
            This allows the encoder to refine the prior's prediction while maintaining the prior's
            influence on the latent space.

            Args:
                input_dict: Dictionary containing model inputs

            Returns:
                Tuple containing:
                - action: The output action after passing through trunk network
                - prior_out: Output dictionary from prior network with mu/logvar
                - encoder_out: Output dictionary from encoder network with mu (as residual) and logvar
            """
            obs = input_dict["obs"].detach().clone()
            mask = input_dict["mask"].detach().clone()
            gt = obs[
                ...,
                obs.shape[-1]
                - mask.shape[-1] : obs.shape[-1]
                - mask.shape[-1]
                + self.input_shape_goal,
            ]
            gt[..., 195:421] = 0
            prior_out = self._prior(input_dict)
            encoder_out = self._encoder(input_dict)
            mu = prior_out["mu"] + encoder_out["mu"]
            logvar = encoder_out["logvar"]
            if "vae_noise" not in input_dict:
                input_dict["vae_noise"] = torch.randn_like(mu)
            z = self.reparameterization(
                mu, torch.exp(0.5 * logvar), input_dict["vae_noise"]
            )
            input_dict["vae_latent"] = z
            action, prediction = self._trunk(input_dict)
            return (
                action,
                {
                    "prior_out": prior_out,
                    "encoder_out": encoder_out,
                    "prediction": prediction,
                    "gt": gt,
                },
            )

    def build(self, name, **kwargs):
        net = InterPriorNetworkBuilder.Network(self.params, **kwargs)
        return net
