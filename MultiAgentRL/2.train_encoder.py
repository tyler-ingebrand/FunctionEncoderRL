import argparse
import os
import time
from copy import deepcopy
from typing import Optional, Tuple, List

import gymnasium as gym
import numpy as np
import torch
from pettingzoo.classic import tictactoe_v3
from tianshou.utils.net.continuous import Critic, ActorProb
from torch import nn
from torch.distributions import Independent, Normal
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer, Batch
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import (
    BasePolicy,
    # DQNPolicy,
    PPOPolicy,
    MultiAgentPolicyManager,
    RandomPolicy,
)
from tianshou.trainer import OffpolicyTrainer, OnpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net, ActorCritic
from tqdm import trange

from dense_tag import dense_tag_env
from encoder import PolicyEncoder
from fixed_ppo import FixedPPOPolicyWrapper
from video_recorder import VideoRecorder


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--eps-test', type=float, default=0.05)
    parser.add_argument('--eps-train', type=float, default=0.1)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument(
        '--gamma', type=float, default=0.9, help='a smaller gamma favors earlier win'
    )
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--target-update-freq', type=int, default=320)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--step-per-epoch', type=int, default=1000)
    parser.add_argument('--step-per-collect', type=int, default=2000)
    parser.add_argument('--update-per-step', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument(
        '--hidden-sizes', type=int, nargs='*', default=[128, 128, 128, 128]
    )
    parser.add_argument('--training-num', type=int, default=10)
    parser.add_argument('--test-num', type=int, default=10)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.1)
    parser.add_argument('--embed-dim', type=int, default=20)
    parser.add_argument(
        '--win-rate',
        type=float,
        default=0.6,
        help='the expected winning rate: Optimal policy can get 0.7'
    )
    parser.add_argument(
        '--watch',
        default=False,
        action='store_true',
        help='no training, '
        'watch the play of pre-trained models'
    )
    # parser.add_argument(
    #     '--agent-id',
    #     type=int,
    #     default=2,
    #     help='the learned agent plays as the'
    #     ' agent_id-th player. Choices are 1 and 2.'
    # )
    parser.add_argument(
        '--resume-path',
        type=str,
        default='',
        help='the path of agent pth file '
        'for resuming from a pre-trained agent'
    )
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--vf-coef", type=float, default=0.25)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--eps-clip", type=float, default=0.2)
    parser.add_argument("--dual-clip", type=float, default=None)
    parser.add_argument("--value-clip", type=int, default=0)
    parser.add_argument("--norm-adv", type=int, default=0)
    parser.add_argument("--recompute-adv", type=int, default=1)
    parser.add_argument("--repeat-per-collect", type=int, default=10)
    # parser.add_argument("--step-per-collect", type=int, default=2048)
    return parser

def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]

def get_encodings(encoder, runners, env):
    encodings = []
    state_space = env.observation_space
    states = state_space.sample(N=10_000)
    states = torch.tensor(states, device=encoder.device)
    for runner in runners:
        actions = runner.forward(Batch(obs=states)).act

def make_ppo_agent(args, action_space, optim=None):
    # model
    net_a = Net(
        args.state_shape,
        hidden_sizes=args.hidden_sizes,
        activation=nn.Tanh,
        device=args.device,
    )
    actor = ActorProb(
        net_a,
        args.action_shape,
        unbounded=True,
        device=args.device,
    ).to(args.device)
    net_c = Net(
        args.state_shape,
        hidden_sizes=args.hidden_sizes,
        activation=nn.Tanh,
        device=args.device,
    )
    critic = Critic(net_c, device=args.device).to(args.device)
    actor_critic = ActorCritic(actor, critic)

    torch.nn.init.constant_(actor.sigma_param, -0.5)
    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            # orthogonal initialization
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)
    # do last policy layer scaling, this will make initial actions have (close to)
    # 0 mean and std, and will help boost performances,
    # see https://arxiv.org/abs/2006.05990, Fig.24 for details
    for m in actor.mu.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.zeros_(m.bias)
            m.weight.data.copy_(0.01 * m.weight.data)

    optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)

    # lr_scheduler = None
    # if args.lr_decay:
    #     # decay learning rate to 0 linearly
    #     max_update_num = np.ceil(args.step_per_epoch / args.step_per_collect) * args.epoch
    #
    #     lr_scheduler = LambdaLR(optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num)

    def dist(*logits):
        return Independent(Normal(*logits), 1)

    policy = PPOPolicy(
        actor,
        critic,
        optim,
        dist,
        discount_factor=args.gamma,
        gae_lambda=args.gae_lambda,
        max_grad_norm=args.max_grad_norm,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        # reward_normalization=args.rew_norm,
        action_scaling=True,
        # action_bound_method=args.bound_action_method,
        # lr_scheduler=lr_scheduler,
        action_space=action_space,
        eps_clip=args.eps_clip,
        value_clip=args.value_clip,
        dual_clip=args.dual_clip,
        advantage_normalization=args.norm_adv,
        recompute_advantage=args.recompute_adv,
    )
    return policy

def get_env(render_mode=None, video_dir=None):
    style = "favors_runner"
    if render_mode == "video":
        path = video_dir
        os.makedirs(path, exist_ok=True)
        return PettingZooEnv(VideoRecorder( dense_tag_env(render_mode="rgb_array", num_good=1, num_adversaries=1, num_obstacles=0, continuous_actions=True, style=style), path))
    else:
        return PettingZooEnv(dense_tag_env(render_mode=render_mode, num_good=1, num_adversaries=1, num_obstacles=0, continuous_actions=True, style=style))


def train_encoder(
    args: argparse.Namespace = get_args(),
    optim: Optional[torch.optim.Optimizer] = None,
) -> None:
    # fetch state and action space info
    throw_away_env = get_env()
    observation_space = throw_away_env.observation_space['observation'] if isinstance(
        throw_away_env.observation_space, gym.spaces.Dict
    ) else throw_away_env.observation_space
    args.state_shape = observation_space.shape or observation_space.n
    args.action_shape = throw_away_env.action_space.shape or throw_away_env.action_space.n

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    # load N runners
    N = 10
    runners = [make_ppo_agent(args, throw_away_env.action_space) for n in range(N)]
    if args.resume_path: # loads previously trained models
        path = args.resume_path
        if path[-6:] != "models":
            path = os.path.join(path, "models")
        for index, agent in enumerate(runners):
            agent.load_state_dict(torch.load(os.path.join(path, f"runner_{index}.pt")))
            agent._deterministic_eval = True # make sure policies are in execution  mode so they are deterministic
            agent.training = False
    else:
        raise Exception("No runners to load! This script is only to train a policy encoder based on a pretrained league. Run 1.test_league.py first. ")
    runners = [FixedPPOPolicyWrapper(r) for r in runners]  # freezes the runners


    # ======== tensorboard logging setup =========
    current_date_time_string = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    base_log_path = os.path.join(args.logdir, 'tag', 'encoder', current_date_time_string)
    logger = SummaryWriter(base_log_path)

    # create function encoder
    policy_encoder =  PolicyEncoder(args.state_shape, args.action_shape, args.embed_dim).to(args.device)
    opt = torch.optim.Adam(policy_encoder.parameters(), lr=1e-3)

    # set up sampling space
    # note we are using "favor runner" so runner vel max = 1.1, tagger max = 1.0, assume all agents stay within -1 to 1
    low = torch.tensor([-1.0, -1.0], device=args.device) # note the obs space for a runner is runner_vel(2), runner_pos(2), opponent_pos(2), opponent_vel(2)
    high = torch.tensor([1.0, 1.0], device=args.device)
    num_samples = 10_000

    # train
    for descent_step in trange(10000):
        states = low + torch.rand(num_samples, len(low), device=args.device) * (high - low)

        # Iterate through all runners and accumulate gradients
        for runner in runners: # note this code can be ~10x faster by first calculating actions for all agents, then doing a single backward pass. But who has time for that.
            batch = Batch(obs=states, info={})
            actions = runner.forward(batch).act
            individual_encodings = policy_encoder(states)
            encoding = torch.einsum("bka, ba -> bk", individual_encodings, actions).mean(dim=0) # of size k

            # compute estimation loss
            estimated_actions = torch.einsum("k, bka -> ba", encoding, individual_encodings)
            assert estimated_actions.shape == actions.shape
            estimation_loss = torch.mean((estimated_actions - actions)**2)
            estimation_loss.backward()
        opt.step()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy_encoder.parameters(), 0.1)
        opt.zero_grad()
        with torch.no_grad():
            logger.add_scalar('loss/loss', estimation_loss.item(), descent_step)

    torch.save(policy_encoder.state_dict(), os.path.join(base_log_path, 'policy_encoder.pt'))

# train the agent and watch its performance in a match!
args = get_args()
train_encoder(args)
