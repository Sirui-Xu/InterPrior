"""The released InterPrior policy for the SMPL humanoid."""

import contextlib
import inspect
import io
from pathlib import Path

import torch
from torch import nn
import yaml
from rl_games.algos_torch.models import ModelA2CContinuousLogStd

from .networks import InterPriorNetworkBuilder
from .serialization import load_tensor_file


class InterPriorPolicy(nn.Module):
    """Map raw state/history observations and sparse goals to PD actions.

    Observations: [batch, 8147]. Goal masks: [batch, 2512].
    Actions: [batch, 153], normalized to [-1, 1].
    A zero latent noise vector gives deterministic prior-mean inference.
    """

    observation_dim = 8147
    mask_dim = 2512
    action_dim = 153
    latent_dim = 64

    def __init__(self, checkpoint, device="cpu"):
        super().__init__()
        payload = load_tensor_file(checkpoint)
        params = yaml.safe_load(
            (Path(__file__).parent / "configs/network.yaml").read_text()
        )
        builder = InterPriorNetworkBuilder()
        builder.load(params)
        trunk_dim = payload["model"]["a2c_network.actor_mlp.0.weight"].shape[1]
        with contextlib.redirect_stdout(io.StringIO()):
            network = builder.build(
                "interprior",
                actions_num=self.action_dim,
                input_shape=(trunk_dim,),
                num_seqs=1,
                value_size=1,
            )
            constructor = inspect.signature(ModelA2CContinuousLogStd.Network.__init__)
            if any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in constructor.parameters.values()
            ):
                self.model = ModelA2CContinuousLogStd.Network(
                    network,
                    obs_shape=(self.observation_dim,),
                    normalize_value=False,
                    normalize_input=False,
                    value_size=1,
                )
            else:
                self.model = ModelA2CContinuousLogStd.Network(network)
        self.model.load_state_dict(payload["model"], strict=True)
        statistics = payload["running_mean_std"]
        for name, key in [
            ("observation_mean", "running_mean"),
            ("observation_var", "running_var"),
        ]:
            value = statistics[key].float()
            if value.shape != (self.observation_dim,):
                raise ValueError("Checkpoint normalization has incompatible dimensions")
            self.register_buffer(name, value)
        self.to(device).eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, observations, mask, noise=None):
        device = self.observation_mean.device
        observations = torch.as_tensor(observations, dtype=torch.float32, device=device)
        mask = torch.as_tensor(mask, dtype=torch.float32, device=device)
        if observations.ndim != 2 or observations.shape[1] != self.observation_dim:
            raise ValueError("Expected observations with shape [batch, 8147]")
        batch = observations.shape[0]
        if mask.shape != (batch, self.mask_dim):
            raise ValueError("Expected goal masks with shape [batch, 2512]")
        noise = (
            torch.zeros((batch, self.latent_dim), device=device)
            if noise is None
            else torch.as_tensor(noise, dtype=torch.float32, device=device)
        )
        if noise.shape != (batch, self.latent_dim):
            raise ValueError("Expected persistent latent noise with shape [batch, 64]")
        inputs = {
            "obs": (
                (observations - self.observation_mean)
                / torch.sqrt(self.observation_var + 1e-5)
            ).clamp(-5, 5),
            "mask": mask,
            "vae_noise": noise,
            "with_encoder": False,
            "with_vae": False,
            "is_train": False,
            "rank": 0,
            "prev_actions": None,
            "rnn_states": None,
        }
        means, _ = self.model.a2c_network.act(inputs)
        return means.clamp(-1, 1)
