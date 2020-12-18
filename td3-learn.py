#!/usr/bin/env python3
import numpy as np
import torch
import gym
import argparse
import os

from ac_gym.td3 import TD3, ReplayBuffer, eval_policy

from sys import stdout

def _save(args, avg_reward, evaluations, policy):
    stdout.flush()
    filename = 'td3-%s%+010.3f' % (args.env, avg_reward)
    np.save('./runs/' + filename, evaluations)
    torch.save((policy.get(), args.env, args.nhid) , open('./models/'+filename+'.dat', 'wb'))

def main():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', default='Pendulum-v0', help='OpenAI gym environment name')
    parser.add_argument('--checkpoint', dest='checkpoint', action='store_true', help='Save at each new best')
    parser.add_argument('--nhid', default='64',type=int, help='Number of hidden units')
    parser.add_argument('--maxeps', default=np.inf, type=int, help='Maximum number of episodes')
    parser.add_argument('--target', type=float, default=np.inf, help='Quitting criterion for average reward')
    parser.add_argument('--gamma', default=0.99, help='Discount factor')
    parser.add_argument('--test-iters', default=10, type=float, help='How often (episodes) to test and save best')
    parser.add_argument('--eval-episodes', default=10, type=float, help='How many episodes to evaluate for average')
    parser.add_argument('--start-episodes', default=125, type=int,help='Epsiodes during which initial random policy is used')
    parser.add_argument('--expl-noise', default=0.1, help='Std of Gaussian exploration noise')
    parser.add_argument('--batch-size', default=256, type=int, help='Batch size for both actor and critic')
    parser.add_argument('--tau', default=0.005, help='Target network update rate')
    parser.add_argument('--policy-noise', default=0.2, help='Noise added to target policy during critic update')
    parser.add_argument('--noise-clip', default=0.5, help='Range to clip target policy noise')
    parser.add_argument('--policy-freq', default=2, type=int, help='Frequency of delayed policy updates')
    args = parser.parse_args()

    os.makedirs('./runs', exist_ok=True)
    os.makedirs('./models', exist_ok=True)

    env = gym.make(args.env)

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

    evaluations = []

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_index = 0
    best_reward = None
    just_tested = False

    print('Running %d episodes with random action ...' % args.start_episodes)

    while episode_index < (args.start_episodes+args.maxeps):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if episode_index < args.start_episodes:
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
        if episode_index >= args.start_episodes:
            policy.train(replay_buffer, args.batch_size)

        if done: 

            if not just_tested:
                if episode_index >= args.start_episodes:
                    print('Episode %07d:\treward = %+.3f,\tsteps = %d' % 
                            (episode_index-args.start_episodes+1, episode_reward, episode_timesteps))

            # Reset everything
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_index += 1 
            just_tested = False

        evaluations.append((episode_index+1,episode_reward))

        # Evaluate episode
        #if episode_index >= args.start_episodes and (episode_index-args.start_episodes) %args.test_iters == 0:
        test_index = episode_index - args.start_episodes
        if test_index > 0 and test_index%args.test_iters == 0:

            print('Episode %07d:\ttesting' % (episode_index-args.start_episodes+1))

            avg_reward,_ = eval_policy(policy, env, args.eval_episodes)

            just_tested = True

            if args.checkpoint and (best_reward is None or best_reward < avg_reward):
                if best_reward is not None:
                    print('\n* Best reward updated: %+.3f -> %+.3f *\n' % (best_reward, avg_reward))
                    _save(args, avg_reward, evaluations, policy)
                best_reward = avg_reward

            if avg_reward >= args.target:
                print('Target average reward %f achieved' % args.target)
                break

    # Save final net
    avg_reward,_ = eval_policy(policy, env, args.eval_episodes)
    _save(args, avg_reward, evaluations, policy)

main()
