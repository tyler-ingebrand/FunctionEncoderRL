import torch
import os
from datetime import datetime
import numpy as np
import random
import pickle
import csv
from grid_modules.replay_buffer import ReplayBuffer
from grid_modules.mdp_utils import extract_policy
from discrete_action_robots_modules.models import ForwardMap, BackwardMap
from continuous_world_modules.featurizer import RadialBasisFunction2D

from torch.distributions.cauchy import Cauchy

"""
FB agent

"""

def get_cos_sin_features(obs):
    sin_features = torch.sin(2 * np.pi * obs)
    cos_features = torch.cos(2 * np.pi * obs)
    return torch.cat([sin_features, cos_features], dim=-1)

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.max(np.abs(goal_a - goal_b), axis=-1)


def compute_entropy(policy):
    p_log_p = torch.log(policy) * policy
    return -p_log_p.sum(-1)


def nanmean(v, *args, inplace=False, **kwargs):
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 1e-10
    return v.mean(*args, **kwargs)


class FBAgent:
    def __init__(self, args, env, env_params):
        self.args = args
        self.env = env
        self.env_params = env_params
        self.cauchy = Cauchy(torch.tensor([0.0]), torch.tensor([0.5]))
        self.featuriser = RadialBasisFunction2D(1, 21, 0.05, cuda=args.cuda)
        # self.featuriser.transform = lambda x: get_cos_sin_features(x)
        # create the network
        self.forward_network = ForwardMap(env_params, args.embed_dim)
        self.backward_network = BackwardMap(env_params, args.embed_dim)
        # build up the target network
        self.forward_target_network = ForwardMap(env_params, args.embed_dim)
        self.backward_target_network = BackwardMap(env_params, args.embed_dim)
        # load the weights into the target networks
        self.forward_target_network.load_state_dict(self.forward_network.state_dict())
        self.backward_target_network.load_state_dict(self.backward_network.state_dict())
        # if use gpu
        if self.args.cuda:
            self.forward_network.cuda()
            self.backward_network.cuda()
            self.forward_target_network.cuda()
            self.backward_target_network.cuda()
        # create the optimizer
        f_params = [param for param in self.forward_network.parameters()]
        b_params = [param for param in self.backward_network.parameters()]
        self.f_optim = torch.optim.Adam(f_params, lr=self.args.lr)
        self.b_optim = torch.optim.Adam(b_params, lr=self.args.lr)
        self.fb_optim = torch.optim.Adam(f_params + b_params, lr=self.args.lr)
        # self.backward_optim = torch.optim.Adam(self.backward_network.parameters(), lr=self.args.lr_backward)
        # her sampler

        # create the replay buffer
        self.buffer = ReplayBuffer(self.args.buffer_size)

        if args.save_dir is not None:
            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)

            print(' ' * 26 + 'Options')
            for k, v in vars(self.args).items():
                print(' ' * 26 + k + ': ' + str(v))

            with open(self.args.save_dir + "/arguments.pkl", 'wb') as f:
                pickle.dump(self.args, f)

            with open('{}/score_monitor.csv'.format(self.args.save_dir), "wt") as monitor_file:
                monitor = csv.writer(monitor_file)
                monitor.writerow(['epoch', 'eval', 'dist', 'eval (GPI)', 'dist (GPI)', 'loss', 'entropy'])

    def learn(self):
        """
        train the network

        """
        best_rate = 0
        # start to collect samples
        for epoch in range(self.args.n_epochs):
            for _ in range(self.args.n_cycles):
                for _ in range(self.args.num_rollouts_per_cycle):
                    # reset the rollouts
                    # reset the environment
                    obs = self.env.reset()
                    g = self.env.goal
                    if self.args.w_sampling == 'goal_oriented':
                        g_tensor = self._preproc_g(g)
                        with torch.no_grad():
                            w = self.backward_network(g_tensor)
                    elif self.args.w_sampling == 'uniform_ball':
                        w = self.sample_uniform_ball(1)
                    elif self.args.w_sampling == 'cauchy_ball':
                        w = self.sample_cauchy_ball(1)
                    # start to collect samples
                    for t in range(self.env_params['max_timesteps']):
                        with torch.no_grad():
                            obs_tensor = self._preproc_o(obs)
                            action = self.act_e_greedy(obs_tensor, w, update_eps=self.args.update_eps)
                        # feed the actions into the environment
                        obs_new, reward, done, info = self.env.step(action)
                        # add transition
                        self.buffer.add(obs, g, action, reward, obs_new, done)
                        if done:
                            obs = self.env.reset()
                            g = self.env.goal
                        else:
                            obs = obs_new
                for _ in range(self.args.n_batches):
                    # train the network
                    fb_loss, entropy = self._update_network()
                # soft update
                self._soft_update_target_network(self.forward_target_network, self.forward_network)
                self._soft_update_target_network(self.backward_target_network, self.backward_network)
            # self._hard_update_target_network(self.forward_target_network, self.forward_network)
            # self._hard_update_target_network(self.backward_target_network, self.backward_network)
            # start to do the evaluation
            success_rate, dist = self._eval_agent()
            gpi_success_rate, gpi_dist = self._eval_gpi_agent(num_gpi=self.args.num_gpi)

            print('[{}] epoch is: {}, eval: {:.3f}, dist: {:.3f}, eval (GPI): {:.3f},'
                  'dist (GPI): {:.3f}, loss: {:.3f}, entropy: {:.3f}'.format(datetime.now(), epoch,
                                                                             success_rate, dist,
                                                                             gpi_success_rate, gpi_dist,
                                                                             fb_loss, entropy))

            with open('{}/score_monitor.csv'.format(self.args.save_dir), "a") as monitor_file:
                monitor = csv.writer(monitor_file)
                monitor.writerow([epoch, success_rate, dist, gpi_success_rate, gpi_dist, fb_loss, entropy])
            torch.save([self.forward_network.state_dict(), self.backward_network.state_dict()],
                       os.path.join(self.args.save_dir, 'model.pt'))
            if success_rate > best_rate:
                best_rate = success_rate
                torch.save([self.forward_network.state_dict(), self.backward_network.state_dict()],
                           os.path.join(self.args.save_dir, 'best_model.pt'))


    def sample_uniform_ball(self, n, eps=1e-10):
        gaussian_rdv = torch.FloatTensor(n, self.args.embed_dim).normal_(mean=0, std=1)
        gaussian_rdv /= torch.norm(gaussian_rdv, dim=-1, keepdim=True) + eps
        uniform_rdv = torch.FloatTensor(n, 1).uniform_()
        w = np.sqrt(self.args.embed_dim) * gaussian_rdv * uniform_rdv
        # w = gaussian_rdv * uniform_rdv
        # w = w.repeat(n, 1)
        if self.args.cuda:
            w = w.cuda()
        return w

    def sample_cauchy_ball(self, n, eps=1e-10):
        gaussian_rdv = torch.FloatTensor(n, self.args.embed_dim).normal_(mean=0, std=1)
        gaussian_rdv /= torch.norm(gaussian_rdv, dim=-1, keepdim=True) + eps
        cauchy_rdv = self.cauchy.sample((n, ))
        w = np.sqrt(self.args.embed_dim) * gaussian_rdv * cauchy_rdv
        if self.args.cuda:
            w = w.cuda()
        return w

    # pre_process the inputs
    def _preproc_o(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            obs_tensor = obs_tensor.cuda()
        obs_tensor = self.featuriser.transform(obs_tensor)
        return obs_tensor

    def _preproc_g(self, g):
        g_tensor = torch.tensor(g, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            g_tensor = g_tensor.cuda()
        g_tensor = self.featuriser.transform(g_tensor)
        return g_tensor

    def get_policy(self, w, obs=None, policy_type='boltzmann', temp=1, eps=0.01, target_network=False):
        if target_network:
            f = self.forward_target_network(obs, w)
        else:
            f = self.forward_network(obs, w)
        z = torch.einsum('sda, sd -> sa', f, w)
        return extract_policy(z, policy_type=policy_type, temp=temp, eps=eps)

    def get_gpi_policy(self, w_train, w_eval, obs=None, policy_type='boltzmann', temp=0.1, eps=0.01):
        num_gpi = w_train.shape[0]
        obs_repeat = obs.repeat(1, num_gpi).reshape(num_gpi * self.env.state_space, -1)
        w_eval_repeat = w_eval.repeat(num_gpi * self.env.state_space, 1)
        w_train_repeat = w_train.repeat(self.env.state_space, 1)
        f = self.forward_network(obs_repeat, w_train_repeat)
        z = torch.einsum('sda, sd -> sa', f, w_eval_repeat).reshape(self.env.state_space,
                                                                    num_gpi,
                                                                    self.env.action_space)
        z = z.max(1)[0]
        return extract_policy(z, policy_type=policy_type, temp=temp, eps=eps)

    def act_gpi(self, obs, w_train, w_eval):
        # import pdb
        # pdb.set_trace()
        num_gpi = w_train.shape[0]
        obs_repeat = obs.repeat(num_gpi, 1)
        w_eval_repeat = w_eval.repeat(num_gpi, 1)
        f = self.forward_network(obs_repeat, w_train)
        z = torch.einsum('sda, sd -> sa', f, w_eval_repeat).max(0)[0]
        return z.max(0)[1]

    # Acts based on single state (no batch)
    def act(self, obs, w, target_network=False):
        if target_network:
            f = self.forward_target_network(obs, w)
        else:
            f = self.forward_network(obs, w)
        z = torch.einsum('sda, sd -> sa', f, w)
        # import pdb
        # pdb.set_trace()
        y = z.max(1)[1]
        return y

    # Acts with an epsilon-greedy policy
    def act_e_greedy(self, obs, g, update_eps=0.2):
        return random.randrange(self.env_params['action']) if random.random() < update_eps else self.act(obs, g).item()

    def act_gpi_e_greedy(self, obs, w_train, w_eval, update_eps=0.2):
        return random.randrange(self.env_params['action']) if random.random() < update_eps \
            else self.act_gpi(obs, w_train, w_eval).item()

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    def _hard_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    # update the network
    def _update_network(self):
        # sample the episodes
        transitions = self.buffer.sample(self.args.batch_size)
        other_transitions = self.buffer.sample(self.args.batch_size)

        # transfer them into the tensor
        obs_tensor = torch.tensor(transitions['obs'], dtype=torch.float32)
        g_tensor = torch.tensor(transitions['g'], dtype=torch.float32)
        obs_next_tensor = torch.tensor(transitions['obs_next'], dtype=torch.float32)
        actions_tensor = torch.tensor(transitions['action'], dtype=torch.long)
        obs_other_tensor = torch.tensor(other_transitions['obs'], dtype=torch.float32)
        actions_other_tensor = torch.tensor(other_transitions['action'], dtype=torch.long)
        if self.args.cuda:
            obs_tensor = obs_tensor.cuda()
            g_tensor = g_tensor.cuda()
            obs_next_tensor = obs_next_tensor.cuda()
            actions_tensor = actions_tensor.cuda()
            obs_other_tensor = obs_other_tensor.cuda()
            actions_other_tensor = actions_other_tensor.cuda()

        # obs featurization
        feat_tensor = self.featuriser.transform(obs_tensor)
        feat_next_tensor = self.featuriser.transform(obs_next_tensor)
        feat_other_tensor = self.featuriser.transform(obs_other_tensor)

        if self.args.w_sampling == 'goal_oriented':
            with torch.no_grad():
                w = self.backward_network(g_tensor)
                w = w.detach()
        elif self.args.w_sampling == 'uniform_ball':
            w = self.sample_uniform_ball(self.args.batch_size)
        elif self.args.w_sampling == 'cauchy_ball':
            w = self.sample_cauchy_ball(self.args.batch_size)

        # calculate the target Q value function
        with torch.no_grad():
            if self.args.soft_update:
                pi = self.get_policy(w, obs=feat_next_tensor, policy_type='boltzmann', temp=self.args.temp,
                                     target_network=True)
                entropy = nanmean(compute_entropy(pi))
                f_next = torch.einsum('sda, sa -> sd', self.forward_target_network(feat_next_tensor, w), pi)
            else:
                actions_next_tensor = self.act(feat_next_tensor, w, target_network=True)
                next_idxs = actions_next_tensor[:, None].repeat(1, self.args.embed_dim)[:, :, None]
                f_next = self.forward_target_network(feat_next_tensor, w).gather(-1, next_idxs).squeeze() # batch x dim
                entropy = torch.zeros(1)

            b_next = self.backward_target_network(feat_other_tensor)  # batch x dim
            # idxs_other = actions_other_tensor[:, None].repeat(1, self.args.embed_dim)[:, :, None]
            # b_next = self.backward_target_network(obs_other_tensor).gather(-1, idxs_other).squeeze()  # batch x dim
            z_next = torch.einsum('sd, td -> st', f_next, b_next)  # batch x batch
            z_next = z_next.detach()

        # the forward loss
        idxs = actions_tensor[:, None].repeat(1, self.args.embed_dim)[:, :, None]
        f = self.forward_network(feat_tensor, w).gather(-1, idxs).squeeze()
        b = self.backward_network(feat_tensor)
        b_other = self.backward_network(feat_other_tensor)
        # b = self.backward_network(obs_tensor).gather(-1, idxs).squeeze()
        # b_other = self.backward_network(obs_other_tensor).gather(-1, idxs_other).squeeze()
        z_diag = torch.einsum('sd, sd -> s', f, b)  # batch
        z = torch.einsum('sd, td -> st', f, b_other)  # batch x batch
        fb_loss = 0.5 * (z - self.args.gamma * z_next).pow(2).mean() - z_diag.mean()
        # compute orthonormality's regularisation loss
        b_b_other = torch.einsum('sd, xd -> sx', b, b_other)  # batch x batch
        b_b_other_detach = torch.einsum('sd, xd -> sx', b, b_other.detach())  # batch x batch
        b_b_detach = torch.einsum('sd, sd -> s', b, b.detach())  # batch
        reg_loss = (b_b_detach * b_b_other.detach()).mean() - b_b_other_detach.mean()
        fb_loss += self.args.reg_coef * reg_loss

        # update the forward_network
        self.fb_optim.zero_grad()
        fb_loss.backward()
        # clip_grad_norm_(self.forward_network.parameters(), 5)
        self.fb_optim.step()

        return fb_loss.item(), entropy.item()

        # the backward loss
        # f = self.forward_network(obs_tensor, w).gather(-1, idxs).squeeze()
        # f = f.detach()
        # b = self.backward_network(obs_tensor)
        # b_other = self.backward_network(obs_other_tensor)
        # z_diag = torch.einsum('sd, sd -> s', f, b)  # batch
        # z = torch.einsum('sd, td -> st', f, b_other)  # batch x batch
        # b_loss = 0.5 * (z - self.args.gamma * z_next).pow(2).mean() - z_diag.mean()
        # # compute orthonormality's regularisation loss
        # b_b_other = torch.einsum('sd, xd -> sx', b, b_other)  # batch x batch
        # b_b_other_detach = torch.einsum('sd, xd -> sx', b, b_other.detach())  # batch x batch
        # b_b_detach = torch.einsum('sd, sd -> s', b, b.detach())  # batch
        # reg_loss = (b_b_detach * b_b_other.detach()).mean() - b_b_other_detach.mean()
        # b_loss += self.args.reg_coef * reg_loss
        #
        # # update the backward_network
        # self.b_optim.zero_grad()
        # b_loss.backward()
        # clip_grad_norm_(self.backward_network.parameters(), 5)
        # self.b_optim.step()

    # do the evaluation
    def _eval_agent(self):
        total_success_rate = []
        total_dist = []
        for _ in range(self.args.n_test_rollouts):
            obs = self.env.reset()
            g = self.env.goal
            # self.env.set_initial_position(Point(0.2, 0.1))
            # self.env.set_goal(Point(0.9, 0.9))
            # obs = self.env.current_position
            # g = self.env.goal
            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    g_tensor = self._preproc_g(g)
                    w = self.backward_network(g_tensor)
                    obs_tensor = self._preproc_o(obs)
                    action = self.act_e_greedy(obs_tensor, w, update_eps=0.02)
                obs, reward, _, info = self.env.step(action)
                if reward > 0:
                    break
            total_success_rate.append(reward)
            dist = goal_distance(obs, g)
            total_dist.append(dist)

        total_success_rate = np.array(total_success_rate)
        total_success_rate = np.mean(total_success_rate)

        total_dist = np.array(total_dist)
        total_dist = np.mean(total_dist)

        return total_success_rate, total_dist

    def _eval_gpi_agent(self, num_gpi=20):
        total_success_rate = []
        total_dist = []
        for _ in range(self.args.n_test_rollouts):
            obs = self.env.reset()
            g = self.env.goal
            # with GPI
            if self.args.w_sampling == 'goal_oriented':
                transitions = self.buffer.sample(num_gpi)
                g_train = transitions['g']
                g_train_tensor = torch.tensor(g_train, dtype=torch.float32)
                if self.args.cuda:
                    g_train_tensor = g_train_tensor.cuda()
                w_train = self.backward_network(g_train_tensor)
            elif self.args.w_sampling == 'uniform_ball':
                w_train = self.sample_uniform_ball(num_gpi)
            elif self.args.w_sampling == 'cauchy_ball':
                w_train = self.sample_cauchy_ball(num_gpi)

            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    g_tensor = self._preproc_g(g)
                    w = self.backward_network(g_tensor)
                    obs_tensor = self._preproc_o(obs)
                    action = self.act_gpi_e_greedy(obs_tensor, w_train, w, update_eps=0.02)
                obs, reward, _, info = self.env.step(action)
                if reward > 0:
                    break

            total_success_rate.append(reward)
            dist = goal_distance(obs, g)
            total_dist.append(dist)

        total_success_rate = np.array(total_success_rate)
        total_success_rate = np.mean(total_success_rate)

        total_dist = np.array(total_dist)
        total_dist = np.mean(total_dist)

        return total_success_rate, total_dist