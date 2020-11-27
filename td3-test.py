#!/usr/bin/env python3
import numpy as np
import torch
import gym
import argparse

from td3 import TD3
from evaluate import eval_policy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', metavar='FILENAME', help='input file')
    parser.add_argument('--env', default='Pendulum-v0', help='OpenAI gym environment name')
    parser.add_argument('--nhid', default='64',type=int,  help='Number of hidden units')
    parser.add_argument('--record', help='If specified, sets the recording dir')
    parser.add_argument('--seed', default=None, type=int, help='Sets Gym, PyTorch and Numpy seeds')
    args = parser.parse_args()

    env = gym.make(args.env)

    # Set seeds if indicated
    if args.seed is not None:
        env.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    if args.record:
        env = wrappers.Monitor(env, args.record)
        
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    kwargs = {
        'state_dim': state_dim,
        'action_dim': action_dim,
        'max_action': max_action,
    }

    # Target policy smoothing is scaled wrt the action scale
    kwargs['nhid'] = args.nhid
    policy = TD3(**kwargs)

    policy.load(args.filename)

    print(eval_policy(policy, args.env, seed=None, render=True, eval_episodes=1))

if __name__ == '__main__':
    main()
