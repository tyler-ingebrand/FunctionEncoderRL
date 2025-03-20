import warnings
from typing import Sequence, Union, Optional, Type, Dict, Any, Tuple

import numpy as np
import torch
from tianshou.utils.net.common import MLP
from tianshou.utils.net.continuous import SIGMA_MIN, SIGMA_MAX
from torch import nn


class OracleCritic(nn.Module):
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
        input_dim = getattr(preprocess_net, "output_dim", preprocess_net_output_dim) + 100
        self.last = MLP(
            input_dim,  # type: ignore
            self.output_dim,
            hidden_sizes,
            device=self.device,
            linear_layer=linear_layer,
            flatten_input=flatten_input,
        )
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
        with torch.no_grad():
            env_indicies = info.get('env_id', np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])[:obs.shape[0]])  # the very first time, info is empty for some reason,but it is only the 10 envs, not training
            one_hot = torch.zeros(env_indicies.shape[0], 100, device=self.device)
            for abc in range(10):
                one_hot[torch.arange(env_indicies.shape[0]), 10 * env_indicies + abc] = 1            
            obs = torch.concat((obs, one_hot), dim=1)

        # pass through network and get output
        logits, hidden = self.preprocess(obs)
        logits = torch.concat((logits, one_hot), dim=1)
        logits = self.last(logits)
        return logits


class OracleActorProb(nn.Module):
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
        action_shape: Sequence[int],
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
        self.output_dim = int(np.prod(action_shape))
        input_dim = getattr(preprocess_net, "output_dim", preprocess_net_output_dim) + 100 # add 10 for one hot encoding
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


    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Any]:
        """Mapping: obs -> logits -> (mu, sigma)."""
        obs = torch.as_tensor(
            obs,
            device=self.device,
            dtype=torch.float32,
        ).flatten(1)

        # get encoding and add it to obs
        with torch.no_grad():
            env_indicies = info.get('env_id', np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])[:obs.shape[0]])  # the very first time, info is empty for some reason,but it is only the 10 envs, not training
            one_hot = torch.zeros(env_indicies.shape[0], 100, device=self.device)
            for abc in range(10):
                one_hot[torch.arange(env_indicies.shape[0]), 10 * env_indicies + abc] = 1
            obs = torch.concat((obs, one_hot), dim=1)

        logits, hidden = self.preprocess(obs, state)
        logits = torch.concat((logits, one_hot), dim=1)
        mu = self.mu(logits)
        if not self._unbounded:
            mu = self.max_action * torch.tanh(mu)
        if self._c_sigma:
            sigma = torch.clamp(self.sigma(logits), min=SIGMA_MIN, max=SIGMA_MAX).exp()
        else:
            shape = [1] * len(mu.shape)
            shape[1] = -1
            sigma = (self.sigma_param.view(shape) + torch.zeros_like(mu)).exp()

        return (mu, sigma), state
