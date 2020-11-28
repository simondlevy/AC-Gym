from time import time
import gym
from libs import ptan, model
import numpy as np
import torch
import math
import argparse
import os

class Solver:

    def __init__(self, args, device, net_act, net_crt):

        self.args = args
        self.device = device
        self.batch = []
        self.net_act = net_act
        self.net_crt = net_crt

def make_nets(args, env, device):
    net_act = model.ModelActor(env.observation_space.shape[0], env.action_space.shape[0], args.nhid).to(device)
    net_crt = model.ModelCritic(env.observation_space.shape[0], args.nhid).to(device)
    return net_act, net_crt

def make_learn_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', default='Pendulum-v0', help='Environment id')
    parser.add_argument('--nhid', default=64, type=int, help='Hidden units')
    parser.add_argument('--target', type=float, default=np.inf, help='Quitting criterion for average reward')
    parser.add_argument('--cuda', default=False, action='store_true', help='Enable CUDA')
    parser.add_argument('--gamma', default=0.99, help='Discount factor')
    parser.add_argument('--datafile', required=False, help='Name of data file to load')
    parser.add_argument('--maxeps', default=None, type=int, help='Maximum number of episodes')
    parser.add_argument('--maxhrs', default=None, type=float, help='Maximum run-time in hours')
    parser.add_argument('--test-iters', default=100, type=float, help='How often to test and save best')
    return parser

def parse_args(parser, algo):
    args = parser.parse_args()
    device = torch.device('cuda' if args.cuda else 'cpu')
    os.makedirs('./models/', exist_ok=True)
    models_path = './models/' + algo + '-' + args.env
    os.makedirs('./runs/', exist_ok=True)
    runs_path = './runs/' + algo + '-' + args.env
    test_env = gym.make(args.env)
    maxeps = np.inf if args.maxeps is None else args.maxeps
    maxsec = np.inf if args.maxhrs is None else (args.maxhrs * 3600)
    return args, device, models_path, runs_path, test_env, maxeps, maxsec

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


def loop(args, exp_source, solver, maxeps, maxsec, test_env, models_path, runs_path):

    best_reward = None
    tstart = time()

    rewards_steps = None

    evaluations = []

    for step_idx, exp in enumerate(exp_source):

        rewards_steps = exp_source.pop_rewards_steps()

        tcurr = time()

        if (tcurr-tstart) >= maxsec or step_idx == maxeps:
            break
        
        if rewards_steps:
            rewards, steps = zip(*rewards_steps)

        if step_idx % args.test_iters == 0:
            reward, steps = test_net(solver.net_act, test_env, device=solver.device)
            print('Episode %07d done in %.2f sec, reward %.3f, steps %d' % (step_idx, time() - tcurr, reward, steps))
            model_fname = models_path + ('%+.3f_%d.dat' % (reward, step_idx))
            evaluations.append((step_idx+1, reward))
            if best_reward is None or best_reward < reward:
                if best_reward is not None:
                    print('Best reward updated: %.3f -> %.3f' % (best_reward, reward))
                    torch.save(solver.clean(solver.net_act.state_dict()), model_fname)
                best_reward = reward
            if args.target is not None and reward >= args.target:
                print('Target %f achieved; saving %s' % (args.target,model_fname))
                torch.save(solver.clean(solver.net_act.state_dict()), model_fname)
                break

        solver.update(exp, maxeps)

    np.save(runs_path+('' if best_reward is None else ('%f'%best_reward)), evaluations)
