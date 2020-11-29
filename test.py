#!/usr/bin/env python3
import argparse
import time
import pickle

import gym
from gym import wrappers

from libs import model
from libs.td3 import TD3, eval_policy

import numpy as np
import torch

def run_td3(parts, env, nhid, record):

    policy = TD3(
            env.observation_space.shape[0],
            env.action_space.shape[0],
            float(env.action_space.high[0]),
            nhid)

    policy.set(parts)

    return eval_policy(policy, env, seed=None, render=(not record), eval_episodes=1)

def run_other(parts, env, nhid, record):

    net = model.ModelActor(env.observation_space.shape[0], env.action_space.shape[0], nhid)

    net.load_state_dict(parts)

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
        if record is None:
            env.render()
            time.sleep(.02)
        total_reward += reward
        total_steps += 1
        if done:
            break

    return total_reward, total_steps


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('filename', metavar='FILENAME', help='input file')
    parser.add_argument('--record', default=None, help='If specified, sets the recording dir')
    parser.add_argument('--seed', default=None, type=int, help='Sets Gym, PyTorch and Numpy seeds')
    args = parser.parse_args()

    #if args.seed is not None:
    #    env.seed(args.seed)
    #    torch.manual_seed(args.seed)
    #    np.random.seed(args.seed)

    #if args.record:
    #    env = wrappers.Monitor(env, args.record)

    parts, env_name, nhid = pickle.load(open(args.filename, 'rb'))

    env = gym.make(env_name)

    fun = run_td3 if 'td3' in args.filename else run_other

    reward, steps = fun(parts, env, nhid, args.record)

    print('In %d steps we got %.3f reward' % (steps, reward))

    env.close()

if __name__ == '__main__':
    main()
