from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from tianshou.data import Batch, ReplayBuffer
from tianshou.policy import BasePolicy

try:
    from tianshou.env.pettingzoo_env import PettingZooEnv
except ImportError:
    PettingZooEnv = None  # type: ignore
from tianshou.policy.multiagent.mapolicy import MultiAgentPolicyManager


class CustomMultiAgentPolicyManager(MultiAgentPolicyManager):

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indice: np.ndarray
    ) -> Batch:
        """Dispatch batch data from obs.agent_id to every policy's process_fn.

        Save original multi-dimensional rew in "save_rew", set rew to the
        reward of each agent during their "process_fn", and restore the
        original reward afterwards.
        """
        results = {}
        # reward can be empty Batch (after initial reset) or nparray.
        has_rew = isinstance(buffer.rew, np.ndarray)
        if has_rew:  # save the original reward in save_rew
            # Since we do not override buffer.__setattr__, here we use _meta to
            # change buffer.rew, otherwise buffer.rew = Batch() has no effect.
            save_rew, buffer._meta.rew = buffer.rew, Batch()
        for agent, policy in self.policies.items():
            agent_index = np.nonzero(batch.obs.agent_id == agent)[0]
            if len(agent_index) == 0:
                results[agent] = Batch()
                continue

            # this breaks for obs_next, since its offset by 1
            # tmp_batch, tmp_indice = batch[agent_index], indice[agent_index]

            # need to implement the same code, but fixed.
            agent_index_for_net_obs = np.nonzero(batch.obs_next.agent_id == agent)[0]
            tmp_batch = Batch()
            tmp_batch.obs = batch.obs[agent_index]
            tmp_batch.obs_next = batch.obs_next[agent_index_for_net_obs]
            tmp_batch.act = batch.act[agent_index]
            tmp_batch.rew = batch.rew[agent_index]
            tmp_batch.done = np.logical_or(batch.done[agent_index], batch.done[agent_index_for_net_obs])
            tmp_batch.terminated = np.logical_or(batch.terminated[agent_index], batch.terminated[agent_index_for_net_obs])
            tmp_batch.truncated = np.logical_or(batch.truncated[agent_index], batch.truncated[agent_index_for_net_obs])
            tmp_batch.info = batch.info[agent_index]
            tmp_batch.info.truncation = np.logical_or(batch.info.truncation[agent_index], batch.info.truncation[agent_index_for_net_obs])
            tmp_indice = indice[agent_index]




            if has_rew:
                tmp_batch.rew = tmp_batch.rew[:, self.agent_idx[agent]]
                buffer._meta.rew = save_rew[:, self.agent_idx[agent]]
            if not hasattr(tmp_batch.obs, "mask"):
                if hasattr(tmp_batch.obs, 'obs'):
                    tmp_batch.obs = tmp_batch.obs.obs
                if hasattr(tmp_batch.obs_next, 'obs'):
                    tmp_batch.obs_next = tmp_batch.obs_next.obs
            results[agent] = policy.process_fn(tmp_batch, buffer, tmp_indice)
        if has_rew:  # restore from save_rew
            buffer._meta.rew = save_rew
        return Batch(results)

