#!/usr/bin/env python3
import numpy as np
import torch
import gym
from gym import wrappers
import argparse

from td3 import TD3, eval_policy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', metavar='FILENAME', help='input file')
    parser.add_argument('--env', default='Pendulum-v0', help='OpenAI gym environment name')
    parser.add_argument('--nhid', default='256',type=int,  help='Number of hidden units')
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
        
    # Target policy smoothing is scaled wrt the action scale
    policy = TD3(
            env.observation_space.shape[0],
            env.action_space.shape[0],
            float(env.action_space.high[0]),
            args.nhid)

    policy.load(args.filename)

    print(eval_policy(policy, env, seed=None, render=(args.record is None), eval_episodes=1))

if __name__ == '__main__':
    main()
