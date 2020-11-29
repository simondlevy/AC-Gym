#!/usr/bin/env python3
import numpy as np
import torch
import gym
import argparse
import os
import pickle

from libs.td3 import TD3, ReplayBuffer, eval_policy

def eval_policy_learn(policy, env, seed, eval_episodes=10):
    avg_reward,_ = eval_policy(policy, env, seed, eval_episodes)
    print('---------------------------------------')
    print('Evaluation over %d episodes: %+.3f' % (eval_episodes, avg_reward))
    print('---------------------------------------')
    return avg_reward

def main():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', default='Pendulum-v0',             help='OpenAI gym environment name')
    parser.add_argument('--nhid', default='64',type=int,            help='Number of hidden units')
    parser.add_argument('--seed', default=0, type=int,              help='Sets Gym, PyTorch and Numpy seeds')
    parser.add_argument('--start_timesteps', default=25e3, type=int,help='Time steps initial random policy is used')
    parser.add_argument('--eval_freq', default=5e3, type=int,       help='How often (time steps, we evaluate')
    parser.add_argument('--max_timesteps', default=1e6, type=int,   help='Max time steps to run environment')
    parser.add_argument('--expl_noise', default=0.1,                help='Std of Gaussian exploration noise')
    parser.add_argument('--batch_size', default=256, type=int,      help='Batch size for both actor and critic')
    parser.add_argument('--gamma', default=0.99,                    help='Discount factor')
    parser.add_argument('--tau', default=0.005,                     help='Target network update rate')
    parser.add_argument('--policy_noise', default=0.2,              help='Noise added to target policy during critic update')
    parser.add_argument('--noise_clip', default=0.5,                help='Range to clip target policy noise')
    parser.add_argument('--policy_freq', default=2, type=int,       help='Frequency of delayed policy updates')
    parser.add_argument('--target', type=float, default=np.inf,     help='Quitting criterion for average reward')
    args = parser.parse_args()

    print('---------------------------------------')
    print('Env: %s, Seed: %d' % (args.env, args.seed))
    print('---------------------------------------')

    os.makedirs('./runs', exist_ok=True)
    os.makedirs('./models', exist_ok=True)

    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    max_action = float(env.action_space.high[0])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy = TD3(
        state_dim,
        action_dim,
		max_action,
        args.nhid,
		discount=args.gamma,
		tau=args.tau,
		policy_noise=args.policy_noise * max_action,
		noise_clip=args.noise_clip * max_action,
		policy_freq=args.policy_freq)

    replay_buffer = ReplayBuffer(state_dim, action_dim)

    env = gym.make(args.env)
    
    # Evaluate untrained policy
    evaluations = [(1, eval_policy_learn(policy, env, args.seed))]

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(args.max_timesteps)):
        
        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                policy.select_action(np.array(state))
                + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action) 
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done: 
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print('Total T: %5d Episode Num: %5d Episode T: %3d Reward: %+6.3f' % 
                    (t+1, episode_num+1, episode_timesteps, episode_reward))
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1 

        evaluations.append((t+1,episode_reward))

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            avg_reward = eval_policy_learn(policy, env, args.seed)
            filename = 'td3-%s%+f' % (args.env, avg_reward)
            np.save('./runs/' + filename, evaluations)
            pickle.dump((policy.get(), args.env, args.nhid) , open('./models/'+filename, 'wb'))
            if avg_reward >= args.target:
                print('Target average reward %f achieved' % args.target)
                break

if __name__ == '__main__':
    main()
