#!/usr/bin/env python3
import argparse
from gym import wrappers

from lib import model, kfac
from PIL import Image
import os

import numpy as np
import torch
import gym

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', metavar='FILENAME', help='input file')
    parser.add_argument('--env', default='Pendulum-v0', help='Environment name to use')
    parser.add_argument('--nhid', default=64, type=int, help='Hidden units')
    parser.add_argument('--record', help='If specified, sets the recording dir, default=Disabled')
    parser.add_argument('--save', type=int, help='If specified, save every N-th step as an image')
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
        total_reward += reward
        total_steps += 1
        if done:
            break
        if args.save is not None and total_steps % args.save == 0:
            o = env.render('rgb_array')
            img = Image.fromarray(o)
            if not os.path.exists('images'):
                os.mkdir('images')
            img.save('images/img_%05d.png' % total_steps)
    print('In %d steps we got %.3f reward' % (total_steps, total_reward))
    env.close()

if __name__ == '__main__':
    main()
