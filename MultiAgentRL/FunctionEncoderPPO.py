from typing import Any, Dict, List, Optional, Type, Union, Tuple

import numpy as np
import torch
from tianshou.policy.base import _gae_return
from torch import nn

from tianshou.data import Batch, ReplayBuffer, to_torch_as, to_numpy
from tianshou.policy import A2CPolicy, PPOPolicy, BasePolicy
from tianshou.utils.net.common import ActorCritic


class FunctionEncoderPPOPolicy(PPOPolicy):
    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.force_agent = -1

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        if self.force_agent != -1:
            batch.info['env_id'] = np.array([self.force_agent]) # if set, forces to play against this agent. Used for testing.
        out =  super().forward(batch, state, **kwargs)
        return out

    # assumes we only play against 1 agent
    def set_agent(self, agent: int):
        self.force_agent = agent


    # note only the lines self.critic(...) changed beacuse we must pass in info
    # they are marked with this: #  TODO: FunctionEncoder
    # so that they stick out
    def _compute_returns(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        v_s, v_s_ = [], []
        with torch.no_grad():
            for minibatch in batch.split(self._batch, shuffle=False, merge_last=True):
                v_s.append(self.critic(minibatch.obs, info=minibatch.info)) #  TODO: FunctionEncoder
                v_s_.append(self.critic(minibatch.obs_next, info=minibatch.info)) #  TODO: FunctionEncoder
        batch.v_s = torch.cat(v_s, dim=0).flatten()  # old value
        v_s = batch.v_s.cpu().numpy()
        v_s_ = torch.cat(v_s_, dim=0).flatten().cpu().numpy()
        # when normalizing values, we do not minus self.ret_rms.mean to be numerically
        # consistent with OPENAI baselines' value normalization pipeline. Emperical
        # study also shows that "minus mean" will harm performances a tiny little bit
        # due to unknown reasons (on Mujoco envs, not confident, though).
        if self._rew_norm:  # unnormalize v_s & v_s_
            v_s = v_s * np.sqrt(self.ret_rms.var + self._eps)
            v_s_ = v_s_ * np.sqrt(self.ret_rms.var + self._eps)
        unnormalized_returns, advantages = self.compute_episodic_return(
            batch,
            buffer,
            indices,
            v_s_,
            v_s,
            gamma=self._gamma,
            gae_lambda=self._lambda
        )
        if self._rew_norm:
            batch.returns = unnormalized_returns / \
                np.sqrt(self.ret_rms.var + self._eps)
            self.ret_rms.update(unnormalized_returns)
        else:
            batch.returns = unnormalized_returns
        batch.returns = to_torch_as(batch.returns, batch.v_s)
        batch.adv = to_torch_as(advantages, batch.v_s)
        return batch

    def learn(  # type: ignore
        self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:
        losses, clip_losses, vf_losses, ent_losses, cos_sims = [], [], [], [], []
        for step in range(repeat):
            if self._recompute_adv and step > 0:
                batch = self._compute_returns(batch, self._buffer, self._indices)
            for minibatch in batch.split(batch_size, merge_last=True):
                # average_cos_sim = self.get_cos_sim(minibatch)
                # cos_sims.append(average_cos_sim)
                
                # calculate loss for actor
                dist = self(minibatch).dist
                if self._norm_adv:
                    mean, std = minibatch.adv.mean(), minibatch.adv.std()
                    minibatch.adv = (minibatch.adv -
                                     mean) / (std + self._eps)  # per-batch norm
                ratio = (dist.log_prob(minibatch.act) -
                         minibatch.logp_old).exp().float()
                ratio = ratio.reshape(ratio.size(0), -1).transpose(0, 1)
                surr1 = ratio * minibatch.adv
                surr2 = ratio.clamp(
                    1.0 - self._eps_clip, 1.0 + self._eps_clip
                ) * minibatch.adv
                if torch.isnan(surr1).any() or torch.isnan(surr2).any():
                    print("HERE")
                if self._dual_clip:
                    clip1 = torch.min(surr1, surr2)
                    clip2 = torch.max(clip1, self._dual_clip * minibatch.adv)
                    clip_loss = -torch.where(minibatch.adv < 0, clip2, clip1).mean()
                else:
                    clip_loss = -torch.min(surr1, surr2).mean()
                if torch.isnan(clip_loss).any() or clip_loss > 100:
                    print("HERE")
                # calculate loss for critic
                value = self.critic(minibatch.obs, info=minibatch.info).flatten() #  TODO: FunctionEncoder
                if self._value_clip:
                    v_clip = minibatch.v_s + \
                        (value - minibatch.v_s).clamp(-self._eps_clip, self._eps_clip)
                    vf1 = (minibatch.returns - value).pow(2)
                    vf2 = (minibatch.returns - v_clip).pow(2)
                    vf_loss = torch.max(vf1, vf2).mean()
                else:
                    vf_loss = (minibatch.returns - value).pow(2).mean()
                # calculate regularization and overall loss
                ent_loss = dist.entropy().mean()
                loss = clip_loss + self._weight_vf * vf_loss \
                    - self._weight_ent * ent_loss
                self.optim.zero_grad()
                loss.backward()
                if self._grad_norm:  # clip large gradient
                    nn.utils.clip_grad_norm_(
                        self._actor_critic.parameters(), max_norm=self._grad_norm
                    )
                self.optim.step()
                clip_losses.append(clip_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(ent_loss.item())
                losses.append(loss.item())

        return {
            "loss": losses,
            "loss/clip": clip_losses,
            "loss/vf": vf_losses,
            "loss/ent": ent_losses,
            # "loss/cos_sim": cos_sims,
        }

    @staticmethod
    def compute_episodic_return(
        batch: Batch,
        buffer: ReplayBuffer,
        indices: np.ndarray,
        v_s_: Optional[Union[np.ndarray, torch.Tensor]] = None,
        v_s: Optional[Union[np.ndarray, torch.Tensor]] = None,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute returns over given batch.

        Use Implementation of Generalized Advantage Estimator (arXiv:1506.02438)
        to calculate q/advantage value of given batch.

        :param Batch batch: a data batch which contains several episodes of data in
            sequential order. Mind that the end of each finished episode of batch
            should be marked by done flag, unfinished (or collecting) episodes will be
            recognized by buffer.unfinished_index().
        :param numpy.ndarray indices: tell batch's location in buffer, batch is equal
            to buffer[indices].
        :param np.ndarray v_s_: the value function of all next states :math:`V(s')`.
        :param float gamma: the discount factor, should be in [0, 1]. Default to 0.99.
        :param float gae_lambda: the parameter for Generalized Advantage Estimation,
            should be in [0, 1]. Default to 0.95.

        :return: two numpy arrays (returns, advantage) with each shape (bsz, ).
        """
        rew = batch.rew
        if v_s_ is None:
            assert np.isclose(gae_lambda, 1.0)
            v_s_ = np.zeros_like(rew)
        else:
            v_s_ = to_numpy(v_s_.flatten())
            v_s_ = v_s_ * BasePolicy.value_mask(buffer, indices)
        v_s = np.roll(v_s_, 1) if v_s is None else to_numpy(v_s.flatten())

        end_flag = np.logical_or(batch.terminated, batch.truncated)
        end_flag[np.isin(indices, buffer.unfinished_index())] = True
        end_flag = np.logical_or(end_flag, batch.info["truncation"])
        advantage = _gae_return(v_s, v_s_, rew, end_flag, gamma, gae_lambda)
        returns = advantage + v_s
        # normalization varies from each policy, so we don't do it here
        return returns, advantage
    
    def get_cos_sim(self, batch: Batch) -> float:
        
        # for each adersary compute gradients. 
        # then we measure the cos sim between gradients
        gradients = []

        for adv in range(10):
            # filter batch by adv
            adv_batch = batch[batch.info['env_id'] == adv]


            # calculate loss for actor
            dist = self(adv_batch).dist
            if self._norm_adv:
                mean, std = adv_batch.adv.mean(), adv_batch.adv.std()
                adv_batch.adv = (adv_batch.adv -
                                    mean) / (std + self._eps)  # per-batch norm
            ratio = (dist.log_prob(adv_batch.act) -
                        adv_batch.logp_old).exp().float()
            ratio = ratio.reshape(ratio.size(0), -1).transpose(0, 1)
            surr1 = ratio * adv_batch.adv
            surr2 = ratio.clamp(
                1.0 - self._eps_clip, 1.0 + self._eps_clip
            ) * adv_batch.adv
            if torch.isnan(surr1).any() or torch.isnan(surr2).any():
                print("HERE")
            if self._dual_clip:
                clip1 = torch.min(surr1, surr2)
                clip2 = torch.max(clip1, self._dual_clip * adv_batch.adv)
                clip_loss = -torch.where(adv_batch.adv < 0, clip2, clip1).mean()
            else:
                clip_loss = -torch.min(surr1, surr2).mean()
            if torch.isnan(clip_loss).any() or clip_loss > 100:
                print("HERE")
            # calculate loss for critic
            value = self.critic(adv_batch.obs, info=adv_batch.info).flatten() #  TODO: FunctionEncoder
            if self._value_clip:
                v_clip = adv_batch.v_s + \
                    (value - adv_batch.v_s).clamp(-self._eps_clip, self._eps_clip)
                vf1 = (adv_batch.returns - value).pow(2)
                vf2 = (adv_batch.returns - v_clip).pow(2)
                vf_loss = torch.max(vf1, vf2).mean()
            else:
                vf_loss = (adv_batch.returns - value).pow(2).mean()
            # calculate regularization and overall loss
            ent_loss = dist.entropy().mean()
            loss = clip_loss + self._weight_vf * vf_loss \
                - self._weight_ent * ent_loss
            self.optim.zero_grad()
            loss.backward()
            
            # get gradients
            grads = []
            for param in self._actor_critic.parameters():
                grads.append(param.grad.flatten())
            grads = torch.cat(grads)
            gradients.append(grads)

        # remember to zero grad
        self.optim.zero_grad()

        # calculate pairwise cos sim
        cos_sims = []
        for i in range(10):
            for j in range(i+1, 10):
                cos_sims.append(torch.nn.functional.cosine_similarity(gradients[i], gradients[j], dim=0).item())
        return np.mean(cos_sims)