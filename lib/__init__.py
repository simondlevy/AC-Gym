import gym
import ptan
import numpy as np
import torch
import math
import argparse
import os


def make_nets(args, env, device):
    net_act = model.ModelActor(env.observation_space.shape[0], env.action_space.shape[0], args.nhid).to(device)
    net_crt = model.ModelCritic(env.observation_space.shape[0], args.nhid).to(device)
    return net_act, net_crt

def make_learn_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=False, action='store_true', help='Enable CUDA')
    parser.add_argument('--name', required=True, help='Name of the run')
    parser.add_argument('--datafile', required=False, help='Name of data file to load')
    parser.add_argument('--env', default='Pendulum-v0', help='Environment id')
    parser.add_argument('--nhid', default=64, type=int, help='Hidden units')
    parser.add_argument('--maxeps', default=None, type=int, help='Maximum number of episodes')
    parser.add_argument('--maxhrs', default=None, type=float, help='Maximum run-time in hours')
    parser.add_argument('--test-iters', default=100000, type=float, help='How often to test and save best')
    return parser

def parse_args(parser, algo):
    args = parser.parse_args()
    device = torch.device('cuda' if args.cuda else 'cpu')
    save_path = os.path.join('models', algo + '-' + args.name)
    os.makedirs(save_path, exist_ok=True)
    test_env = gym.make(args.env)
    maxeps = np.inf if args.maxeps is None else args.maxeps
    maxsec = np.inf if args.maxhrs is None else (args.maxhrs * 3600)
    return args, device, save_path, test_env, maxeps, maxsec

def test_net(net, env, count=10, device='cpu'):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs = env.reset()
        while True:
            obs_v = ptan.agent.float32_preprocessor([obs]).to(device)
            mu_v = net(obs_v)[0]
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            action = np.clip(action, -1, 1)
            if np.isscalar(action): 
                action = [action]
            obs, reward, done, _ = env.step(action)
            rewards += reward
            steps += 1
            if done:
                break
    return rewards / count, steps / count

def calc_logprob(mu_v, logstd_v, actions_v):
    p1 = - ((mu_v - actions_v) ** 2) / (2*torch.exp(logstd_v).clamp(min=1e-3))
    p2 = - torch.log(torch.sqrt(2 * math.pi * torch.exp(logstd_v)))
    return p1 + p2



