#!/usr/bin/env python3
import argparse
from gym import wrappers

from lib import model
from PIL import Image
import os
import time

import numpy as np
import torch
import gym

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', metavar='FILENAME', help='input file')
    parser.add_argument('--env', default='Pendulum-v0', help='Environment name to use')
    parser.add_argument('--nhid', default=64, type=int, help='Hidden units')
    parser.add_argument('--record', default=None, help='If specified, sets the recording dir')
    parser.add_argument('--seed', default=None, type=int, help='Sets Gym, PyTorch and Numpy seeds')
    args = parser.parse_args()

    env = gym.make(args.env)

    if args.seed is not None:
        env.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    if args.record:
        env = wrappers.Monitor(env, args.record)

    net = model.ModelActor(env.observation_space.shape[0], env.action_space.shape[0], args.nhid)
    net.load_state_dict(torch.load(args.filename))

    obs = env.reset()
    total_reward = 0.0
    total_steps = 0
    while True:
        obs_v = torch.FloatTensor(obs)
        mu_v = net(obs_v)
        action = mu_v.squeeze(dim=0).data.numpy()
        action = np.clip(action, -1, 1)
        if np.isscalar(action): 
            action = [action]
        obs, reward, done, _ = env.step(action)
        if args.record is None:
            env.render()
            time.sleep(.02)
        total_reward += reward
        total_steps += 1
        if done:
            break
    print('In %d steps we got %.3f reward' % (total_steps, total_reward))
    env.close()

if __name__ == '__main__':
    main()
