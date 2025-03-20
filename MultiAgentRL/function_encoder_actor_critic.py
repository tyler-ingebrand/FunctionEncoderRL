import warnings
from typing import Sequence, Union, Optional, Type, Dict, Any, Tuple

import numpy as np
import torch
from tianshou.utils.net.common import MLP
from tianshou.utils.net.continuous import SIGMA_MIN, SIGMA_MAX
from torch import nn


class FunctionEncoderCritic(nn.Module):
    """Simple critic network. Will create an actor operated in continuous \
    action space with structure of preprocess_net ---> 1(q value).

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param int preprocess_net_output_dim: the output dimension of
        preprocess_net.
    :param linear_layer: use this module as linear layer. Default to nn.Linear.
    :param bool flatten_input: whether to flatten input data for the last layer.
        Default to True.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    .. seealso::

        Please refer to :class:`~tianshou.utils.net.common.Net` as an instance
        of how preprocess_net is suggested to be defined.
    """

    def __init__(
        self,
        preprocess_net: nn.Module,
        embed_size:int,
        all_encodings,
        hidden_sizes: Sequence[int] = (),
        device: Union[str, int, torch.device] = "cpu",
        preprocess_net_output_dim: Optional[int] = None,
        linear_layer: Type[nn.Linear] = nn.Linear,
        flatten_input: bool = True,
    ) -> None:
        super().__init__()
        self.device = device
        self.preprocess = preprocess_net
        self.output_dim = 1 # embed_size # 1
        input_dim = getattr(preprocess_net, "output_dim", preprocess_net_output_dim) + embed_size
        self.last = MLP(
            input_dim,  # type: ignore
            self.output_dim,
            hidden_sizes,
            device=self.device,
            linear_layer=linear_layer,
            flatten_input=flatten_input,
        )
        # this part is different from normal
        assert embed_size == all_encodings.shape[1]
        self.embed_size = embed_size
        self.encodings = all_encodings.to(self.device)

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        act: Optional[Union[np.ndarray, torch.Tensor]] = None,
        info: Dict[str, Any] = {},
    ) -> torch.Tensor:
        """Mapping: (s, a) -> logits -> Q(s, a)."""
        obs = torch.as_tensor(
            obs,
            device=self.device,
            dtype=torch.float32,
        ).flatten(1)
        if act is not None:
            act = torch.as_tensor(
                act,
                device=self.device,
                dtype=torch.float32,
            ).flatten(1)
            obs = torch.cat([obs, act], dim=1)

        # get encoding and add it to obs
        env_indicies = info['env_id']
        encodings = self.encodings[env_indicies]
        obs = torch.concat((obs.to(self.encodings.device), encodings), dim=1)

        # pass through network and get output
        logits, hidden = self.preprocess(obs)
        logits = torch.cat((logits, encodings), dim=1)
        logits = self.last(logits)
        # logits = torch.sum(logits * encodings, dim=1)
        return logits


class FunctionEncoderActorProb(nn.Module):
    """Simple actor network (output with a Gauss distribution).

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param action_shape: a sequence of int for the shape of action.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param float max_action: the scale for the final action logits. Default to
        1.
    :param bool unbounded: whether to apply tanh activation on final logits.
        Default to False.
    :param bool conditioned_sigma: True when sigma is calculated from the
        input, False when sigma is an independent parameter. Default to False.
    :param int preprocess_net_output_dim: the output dimension of
        preprocess_net.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    .. seealso::

        Please refer to :class:`~tianshou.utils.net.common.Net` as an instance
        of how preprocess_net is suggested to be defined.
    """

    def __init__(
        self,
        preprocess_net: nn.Module,
        embed_size:int,
        action_shape: Sequence[int],
        all_encodings: torch.Tensor,
        hidden_sizes: Sequence[int] = (),
        max_action: float = 1.0,
        device: Union[str, int, torch.device] = "cpu",
        unbounded: bool = False,
        conditioned_sigma: bool = False,
        preprocess_net_output_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if unbounded and not np.isclose(max_action, 1.0):
            warnings.warn(
                "Note that max_action input will be discarded when unbounded is True."
            )
            max_action = 1.0
        self.preprocess = preprocess_net
        self.device = device
        self.output_dim = int(np.prod(action_shape)) # * embed_size
        input_dim = getattr(preprocess_net, "output_dim", preprocess_net_output_dim) + embed_size
        self.mu = MLP(
            input_dim,  # type: ignore
            self.output_dim,
            hidden_sizes,
            device=self.device
        )
        self._c_sigma = conditioned_sigma
        if conditioned_sigma:
            self.sigma = MLP(
                input_dim,  # type: ignore
                self.output_dim,
                hidden_sizes,
                device=self.device
            )
        else:
            self.sigma_param = nn.Parameter(torch.zeros(int(np.prod(action_shape)), 1))
        self.max_action = max_action
        self._unbounded = unbounded

        # this part is different from normal
        assert embed_size == all_encodings.shape[1], f"Embed size {embed_size} does not match encoding size {all_encodings.shape[1]}"
        self.embed_size = embed_size
        self.encodings = all_encodings.to(self.device)
        print(self.encodings)

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Any]:
        """Mapping: obs -> logits -> (mu, sigma)."""
        assert torch.isnan(torch.tensor(obs)).any() == False
        env_indicies = info.get('env_id', np.array([0,1,2,3,4,5,6,7,8,9])[:obs.shape[0]]) # the very first time, info is empty for some reason,but it is only the 10 envs, not training
        assert torch.isnan(torch.tensor(env_indicies)).any() == False
        encodings = self.encodings[env_indicies]
        obs = torch.concat((torch.tensor(obs, device=encodings.device), encodings), dim=1)
        logits, hidden = self.preprocess(obs, state)
        logits = torch.cat((logits, encodings), dim=1)
        mu = self.mu(logits)
        # mu = mu.reshape(mu.shape[0], -1, self.embed_size)
        # mu = torch.sum(mu * encodings.unsqueeze(1), dim=2)
        if not self._unbounded:
            mu = self.max_action * torch.tanh(mu)
        if self._c_sigma:
            sigma = torch.clamp(self.sigma(logits), min=SIGMA_MIN, max=SIGMA_MAX).exp()
        else:
            shape = [1] * len(mu.shape)
            shape[1] = -1
            sigma = (self.sigma_param.view(shape) + torch.zeros_like(mu)).exp()
        if torch.isnan(mu).any() or torch.isnan(sigma).any():
            print("NANS")

        return (mu, sigma), state
