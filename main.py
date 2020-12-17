import copy
import glob
import os
import time
from collections import deque

# TODO remove warning for tensorflow and for pytorch nonzero

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate

import argparse

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--algo', default='a2c', help='algorithm to use: a2c | ppo ')
    parser.add_argument(
        '--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer alpha (default: 0.99)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--use-gae',
        action='store_true',
        default=True,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.01,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--training',
        action='store_true',
        default=False,
        help="Whether or not to train the model")
    parser.add_argument(
        '--save-experimental-data',
        action='store_true',
        default=False,
        help="Whether or not to save experimental data for analysis")
    parser.add_argument(
        '--num-processes',
        type=int,
        default=32,
        help='how many training CPU processes to use (default: 16)')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=40,
        help='number of forward steps in A2C (default: 5)')
    parser.add_argument(
        '--shift-len',
        type=int,
        default=10,
        help='number of steps to shift the rollout forward each update' +
             ' (default: 10)')
    parser.add_argument(
        '--ppo-epoch',
        type=int,
        default=4,
        help='number of ppo epochs (default: 4)')
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=32,
        help='number of batches for ppo (default: 32)')
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.2,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=100,
        help='save interval, one save per n updates (default: 100)')
    parser.add_argument(
        '--load-id',
        type=str,
        help='The ID of the model to load (only if a model is to be loaded)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=None,
        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=10e8,
        help='number of environment steps to train (default: 10e8)')
    parser.add_argument(
        '--env-name',
        default="Bandit-v0", # Or try "Pendulum-v0" | 'CartPole-v0' etc.
        help='environment to train on (default: Bandit-v0)')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument(
        '--use-proper-time-limits',
        action='store_true',
        default=True,
        help='compute returns taking into account time limits')
    parser.add_argument(
        '--recurrent-policy',
        action='store_true',
        default=True,
        help='use a recurrent policy')
    parser.add_argument(
        '--use-linear-lr-decay',
        action='store_true',
        default=False,
        help='use a linear schedule on the learning rate')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    assert args.algo in ['a2c', 'ppo']

    return args


def main():
    args = get_args()

    if not args.training:
        print("Warning! Training is turned off!")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # Set up the directories and names for saving stuff
    session_name = utils.datetimenow(subseconds=True)
    if args.load_id:
        loading_id = args.load_id
        cutoff_idx = loading_id.find('_s')
        model_name = loading_id[1:cutoff_idx]
    else:
        model_name = session_name  #i.e. if new model, model_name is same as session_name

    unique_id = 'm'+ model_name + '_s' + session_name

    print("The unique ID for this model and session combination " + \
          "is %s" % str(unique_id))

    # Make dirs to log experimental data, models
    exp_dir = '../exps'
    if not os.path.isdir(exp_dir):
        os.mkdir(exp_dir)

    data_logs_dir = os.path.join(exp_dir, 'data_logs')
    if not os.path.isdir(data_logs_dir):
        os.mkdir(data_logs_dir)

    data_logs_dir_uniq = os.path.join(data_logs_dir, unique_id)
    if not os.path.isdir(data_logs_dir_uniq):
        os.mkdir(data_logs_dir_uniq)
        print("Data will be logged to %s. " % data_logs_dir_uniq + \
              "Ignore the tmp/openai logging statement below.")

    models_dir = os.path.join(exp_dir, 'models')
    if not os.path.isdir(models_dir):
        os.mkdir(models_dir)
        print("New model will be saved at %s" % models_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    # Save args into csv for record keeping
    utils.save_configs_to_csv(args, session_name=session_name,
                              model_name=model_name, unique_id=unique_id)

    # Set up envs and model etc
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, data_logs_dir_uniq, device, False)

    if args.load_id:
        loaded_id = str(args.load_id)
        path = '../exps/models/' + loaded_id + '.pt'
        actor_critic = torch.load(path)
    else:
        actor_critic = Policy(
            obs_shape=envs.observation_space.shape,
            action_space=envs.action_space,
            base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C(
            args,
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            args, # todo change ppo script to include args (which was added
            # to help make the training fully episodic and with overlapping
            # segments)
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)

    rollouts = RolloutStorage(200, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_episodes = int(10e9)

    for j in range(num_episodes):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_episodes, args.lr)

        for step in range(200):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states, internals  = \
                    actor_critic.act(rollouts.obs[step],
                                     rollouts.recurrent_hidden_states[step],
                                     rollouts.masks[step])

            # Obser reward and next obs
            # envs.render()
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():  # Only added at the end of an epi
                    episode_rewards.append(info['episode']['r'])

            p_dists = torch.FloatTensor(
                [info['p_dist'] if 'p_dist' in info.keys() else np.zeros(2)
                 for info in infos])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])

            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks,
                            bad_masks, p_dists)#TODOnow internals

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        if args.training:
            value_loss, action_loss, dist_entropy = agent.update(rollouts)

        if args.save_experimental_data:
            rollouts.save_experimental_data(save_dir=data_logs_dir_uniq)

        if "Bandit" in args.env_name:
            reset_hxs_every_episode = True
        else:
            reset_hxs_every_episode = False
        rollouts.after_update(reset_hxs_every_episode)

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0 or j == num_episodes - 1):
            torch.save(actor_critic,
                       os.path.join(models_dir, unique_id + ".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1 and args.training:
            total_num_steps = (j + 1) * args.num_processes * 200
            end = time.time()
            print(
                "Episodes {}, num timesteps {}, FPS {}. Entropy: {:.4f} , Value loss: {:.4f}, Policy loss: {:.4f}, \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)), dist_entropy,
                        value_loss, action_loss,
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards)))
        elif j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * 200
            end = time.time()
            print(
                "Episodes {}, num timesteps {}. \n"
                .format(j, total_num_steps))

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(actor_critic, ob_rms, args.env_name, args.seed,
                     args.num_processes, data_logs_dir_uniq, device)


if __name__ == "__main__":
    main()
