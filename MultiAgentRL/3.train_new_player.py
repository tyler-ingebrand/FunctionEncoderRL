import argparse
import os
import time
from copy import deepcopy
from typing import Optional, Tuple, List, Any

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

from FunctionEncoderPPO import FunctionEncoderPPOPolicy
from MyPPO import MyPPOPolicy
from TransformerPPO import TransformerPPOPolicy
from dense_tag import dense_tag_env
from encoder import PolicyEncoder
from fixed_ppo import FixedPPOPolicyWrapper
from function_encoder_actor_critic import FunctionEncoderCritic, FunctionEncoderActorProb
from league_policy import LeaguePolicy
from oracle_actor_critic import OracleCritic, OracleActorProb
from transformer_actor_critic import TransformerActorProb, TransformerCritic
from video_recorder import VideoRecorder
from CustomMultiAgentPolicyManager import CustomMultiAgentPolicyManager

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
    parser.add_argument('--encoder-dir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.1)
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
    parser.add_argument('--embed-dim', type=int, default=20)
    parser.add_argument('--alg-type', type=str, default="FE_PPO")  ## Either PPO or FE_PPO

    # parser.add_argument("--step-per-collect", type=int, default=2048)
    return parser

def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]

def get_encodings(encoder, runners, args, print_similarities=True):
    with torch.no_grad():
        encodings = []
        batches = []
        low = torch.tensor([-1.0, -1.0,], device=args.device)  # note the obs space for a runner is runner_vel(2), runner_pos(2), opponent_pos(2), opponent_vel(2)
        high = torch.tensor([1.0, 1.0,], device=args.device)
        num_samples = 10_000
        states = low + torch.rand(num_samples, len(low), device=args.device) * (high - low)
        for runner in runners:
            batch = Batch(obs=states, info={})
            actions = runner.forward(batch).act
            individual_encodings = encoder(states)
            encoding = torch.einsum("bka, ba -> bk", individual_encodings, actions).mean(dim=0)  # of size k
            encodings.append(encoding)
            batch.act = actions
            batches.append(batch)
        if print_similarities:
            for i in range(len(encodings)):
                for j in range(len(encodings)):
                    cos_sim = torch.nn.functional.cosine_similarity(encodings[i].flatten(), encodings[j].flatten(), dim=0)
                    print(f"cosine similarity between runner {i} and runner {j}: {cos_sim}")

        return encodings, batches


def make_function_encoder_ppo_agent(args, action_space, encodings, optim=None):
    # model
    num_actions = action_space.low.shape[0]
    net_a = Net(
        args.state_shape[0] + args.embed_dim,
        hidden_sizes=args.hidden_sizes,
        activation=nn.Tanh,
        device=args.device,
    )
    actor = FunctionEncoderActorProb(
        net_a,
        args.embed_dim ,
        args.action_shape,
        all_encodings=encodings,
        unbounded=True,
        device=args.device,
    ).to(args.device)
    net_c = Net(
        args.state_shape[0] + args.embed_dim,
        hidden_sizes=args.hidden_sizes,
        activation=nn.Tanh,
        device=args.device,
    )
    critic = FunctionEncoderCritic(net_c,
                                   args.embed_dim,
                                   all_encodings=encodings,
                                   device=args.device).to(args.device)
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

    policy = FunctionEncoderPPOPolicy(
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

    policy = MyPPOPolicy(
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

def make_oracle_ppo_agent(args, action_space, optim=None):
    # model
    args.state_shape = (args.state_shape[0] + 100,) # 10 agents in league
    net_a = Net(
        args.state_shape,
        hidden_sizes=args.hidden_sizes,
        activation=nn.Tanh,
        device=args.device,
    )
    actor = OracleActorProb(
        net_a,
        args.action_shape,
        unbounded=True,
        device=args.device,
    ).to(args.device)
    net_c = Net(
        args.state_shape ,
        hidden_sizes=args.hidden_sizes,
        activation=nn.Tanh,
        device=args.device,
    )
    critic = OracleCritic(net_c, device=args.device).to(args.device)
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

    policy = FunctionEncoderPPOPolicy( # note we do not use a function encoder here, but we do need to pass info into vlaue function
        actor,                  # so we can reuse this class.
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

def make_transformer_ppo_agent(args, state_space, action_space, example_data, optim=None):
    # turn batch example data into tensor
    states = torch.zeros(len(example_data), *example_data[0].obs.shape, device=args.device, requires_grad=False)
    actions = torch.zeros(len(example_data), *example_data[0].act.shape, device=args.device, requires_grad=False)
    for b in range(len(example_data)):
        states[b] = example_data[b].obs
        actions[b] = example_data[b].act
    example_data  = (states, actions)

    # model
    actor = TransformerActorProb(
        state_space,
        action_space,
        example_data=example_data,
        unbounded=True,
        device=args.device,
    ).to(args.device)
    critic = TransformerCritic(state_space=state_space, action_space=action_space, example_data=example_data, device=args.device).to(args.device)
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
    for m in actor.decoder.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.zeros_(m.bias)
            m.weight.data.copy_(0.01 * m.weight.data)
    # UPDATE: note we divide by 5 here because the batch size in learn is 100, which is
    # apporximately 1/5 of the batch size in the other agents.
    # therefore, PPO fails if we dont do this because the policy deviates too far from the initial policy every
    # time we call learn. So, we fix this by making step size smaller (hopefully).
    # UPDATE 2: We further decrease lr because this proved helpful for the multi-task example,
    # however it does not work in this case
    optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr / 100.)

    # lr_scheduler = None
    # if args.lr_decay:
    #     # decay learning rate to 0 linearly
    #     max_update_num = np.ceil(args.step_per_epoch / args.step_per_collect) * args.epoch
    #
    #     lr_scheduler = LambdaLR(optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num)

    def dist(*logits):
        return Independent(Normal(*logits), 1)

    policy = TransformerPPOPolicy( # note we also need to pass env ids in, so we use FunctionEncoderPPOPolicy
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

def get_agents(
    args: argparse.Namespace = get_args(),
    agent_1: BasePolicy = None,
    agent_2: BasePolicy = None,
    optim: Optional[torch.optim.Optimizer] = None,
) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
    env = get_env()
    observation_space = env.observation_space['observation'] if isinstance(
        env.observation_space, gym.spaces.Dict
    ) else env.observation_space
    args.state_shape = observation_space.shape or observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    # if agent_1 is None:
    #     if args.resume_path:
    #         agent_1.load_state_dict(torch.load(args.resume_path))
    #     agent_1 = make_ppo_agent(args, env.action_space)
    #
    # if agent_2 is None:
    #     if args.opponent_path:
    #         agent_2 = deepcopy(agent_1)
    #         agent_2.load_state_dict(torch.load(args.opponent_path))
    #     else:
    #         agent_2 = make_ppo_agent(args, env.action_space)

    agents = [agent_1, agent_2]

    policy = CustomMultiAgentPolicyManager(agents, env)

    return policy, optim, env.agents

def get_env(render_mode=None, video_dir=None, max_cycles=50):
    style = "favors_runner"
    if render_mode == "video":
        path = video_dir
        os.makedirs(path, exist_ok=True)
        return PettingZooEnv(VideoRecorder( dense_tag_env(render_mode="rgb_array", num_good=1, num_adversaries=1, num_obstacles=0, continuous_actions=True, style=style, max_cycles=max_cycles), path))
    else:
        return PettingZooEnv(dense_tag_env(render_mode=render_mode, num_good=1, num_adversaries=1, num_obstacles=0, continuous_actions=True, style=style, max_cycles=max_cycles))

def save_agents(tagger, runners, base_log_path):
    path = os.path.join(base_log_path, "models")
    os.makedirs(path, exist_ok=True)
    torch.save(tagger.state_dict(), os.path.join(path, f"tagger.pt"))

def train_league(
    args: argparse.Namespace = get_args(),
    optim: Optional[torch.optim.Optimizer] = None,
) -> Tuple[Any, Any, str, List]:

    assert args.alg_type == "FE_PPO" or args.alg_type == "PPO" or args.alg_type == "Transformer_PPO" or args.alg_type == "Oracle_PPO", "alg_type must be either FE_PPO or PPO or Transformer_PPO or Oracle_PPO"
    # fetch state and action space info
    throw_away_env = get_env()
    observation_space = throw_away_env.observation_space['observation'] if isinstance(
        throw_away_env.observation_space, gym.spaces.Dict
    ) else throw_away_env.observation_space
    args.state_shape = observation_space.shape or observation_space.n
    args.action_shape = throw_away_env.action_space.shape or throw_away_env.action_space.n


    # ======== environment setup =========
    N = args.training_num
    train_envs = DummyVectorEnv([get_env for _ in range(N)])
    test_envs = DummyVectorEnv([get_env for _ in range(N)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # load encoder
    encoder = PolicyEncoder(args.state_shape, args.action_shape, args.embed_dim)
    encoder.load_state_dict(torch.load(os.path.join(args.encoder_dir, "policy_encoder.pt")))
    encoder.to(args.device)

    # load N runners
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
        raise Exception("No runners to load! This script is only to train a new agent based on a pretrained league. Run 1.test_league.py and 2.train_encoder.py first. ")
    runners = [FixedPPOPolicyWrapper(r) for r in runners]  # freezes the runners
    league_policy = LeaguePolicy(runners)

    encodings, example_data = get_encodings(encoder, runners, args)
    encodings = torch.stack(encodings)
    encodings = encodings * (args.embed_dim**0.5 / torch.mean(torch.norm(encodings, dim=1)))

    if args.alg_type == "FE_PPO":
        tagger = make_function_encoder_ppo_agent(args, throw_away_env.action_space, encodings)
    elif args.alg_type == "Transformer_PPO":
        tagger = make_transformer_ppo_agent(args, args.state_shape, throw_away_env.action_space, example_data)
    elif args.alg_type == "Oracle_PPO":
        tagger = make_oracle_ppo_agent(args, throw_away_env.action_space)
    else:
        tagger = make_ppo_agent(args, throw_away_env.action_space)

    # ======== tensorboard logging setup =========
    current_date_time_string = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    base_log_path = os.path.join(args.logdir, 'tag', 'new_players', current_date_time_string)
    writer = SummaryWriter(base_log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer, update_interval=1)

    # create a tag to save algorithm type in the file dir
    if args.alg_type == "FE_PPO":
        open(os.path.join(base_log_path, "function_encoder_ppo.txt"), "a").close()
    elif args.alg_type == "Transformer_PPO":
        open(os.path.join(base_log_path, "transformer_ppo.txt"), "a").close()
    elif args.alg_type == "Oracle_PPO":
        open(os.path.join(base_log_path, "oracle_ppo.txt"), "a").close()
    else:
        open(os.path.join(base_log_path, "normal_ppo.txt"), "a").close()

    open(os.path.join(base_log_path, f"seed_{args.seed}.txt"), "a").close() # also write seed

    # reusable variables
    buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))

    # ======== agent setup =========
    policy, optim, agents = get_agents(
        args, agent_1=tagger, agent_2=league_policy, optim=optim
    )

    # ======== collector setup =========
    # note this is needed to tell the tagger when the episode ends. Since the runner goes last
    # the tagger never sees the end of the episode otherwise, which causes the value to propagate between
    # episodes, which breaks everything.
    def collector_preprocess_function(info, **kwargs):
        truncated = [env_info["truncation"] for env_info in info]
        return Batch(truncated=truncated)


    train_collector = Collector(
        policy,
        train_envs,
        buffer,
        # preprocess_fn=collector_preprocess_function,
        exploration_noise=True
    )
    # policy.set_eps(1)
    train_collector.collect(n_step=args.batch_size * args.training_num)
    test_collector = Collector(policy,
                               test_envs,
                               # preprocess_fn=collector_preprocess_function,
                               exploration_noise=True)

    def reward_metric(rews): # this is only for tensorboard purposes. We want to see the runners score to see if they are evading or not.
        return rews[:, 0]

    # trainer
    result = OnpolicyTrainer(
        policy,
        train_collector,
        test_collector,
        max_epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        repeat_per_collect=args.repeat_per_collect,
        # args.step_per_collect,
        episode_per_test=args.test_num,
        batch_size=args.batch_size,
        # train_fn=train_fn,
        # test_fn=test_fn,
        step_per_collect=args.step_per_collect,
        # stop_fn=stop_fn,
        # save_best_fn=save_best_fn,
        update_per_step=args.update_per_step,
        logger=logger,
        test_in_train=False,
        reward_metric=reward_metric,
        show_progress=True,
        verbose=False,
    ).run()

    # result = test_collector.collect(n_episode=10, render=args.render)
    # mean_tagger_score = result["rews"][:, 0].mean()

    # save models
    save_agents(tagger, runners, base_log_path)
    return tagger, league_policy, base_log_path, encodings

    # ======== a test function that tests a pre-trained agent ======
def watch(
    args: argparse.Namespace = get_args(),
    tagger: BasePolicy = None,
    runner: BasePolicy = None,
    video_dir: Optional[str] = None,
    encodings: Optional[List[torch.Tensor]] = None,
) -> None:
    video_dir = os.path.join(video_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)

    env = get_env(render_mode="video", video_dir=video_dir, max_cycles=100)
    env = DummyVectorEnv([lambda: env])

    policy, optim, agents = get_agents(
        args, agent_1=tagger, agent_2=runner, optim=None
    )
    policy.eval()
    # policy.policies[agents[args.agent_id - 1]].set_eps(args.eps_test)
    collector = Collector(policy, env, exploration_noise=True)
    for i in range(10):
        runner.set_agent(i)
        if hasattr(tagger, "set_agent"):
            tagger.set_agent(i)
        result = collector.collect(n_episode=1, render=args.render)
    # rews, lens = result["rews"], result["lens"]
    # print(f"Final reward: {rews}")
    # print(f"Final reward: {rews[:, 0].mean()}, length: {lens.mean()}")
    # print(f"Final reward: {rews[:, 1].mean()}, length: {lens.mean()}")

# train the agent and watch its performance in a match!
args = get_args()
taggers, runners, base_log_path, encodings = train_league(args)
watch(args, taggers, runners,video_dir=base_log_path, encodings=encodings)
