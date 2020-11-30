from time import time
import gym
from libs import ptan, model
import numpy as np
import torch
import argparse
import os
import torch.optim as optim

class Solver:

    def __init__(self, env_name, algo_name, nhid, cuda, gamma, lr_critic):

        self.env = gym.make(env_name)

        os.makedirs('./models/', exist_ok=True)
        self.models_path = './models/' + algo_name + '-' + env_name
        os.makedirs('./runs/', exist_ok=True)
        self.runs_path = './runs/' + algo_name + '-' + env_name

        self.device = torch.device('cuda' if cuda else 'cpu')
        self.env_name = env_name
        self.nhid = nhid

        self.net_act = model.ModelActor(self.env.observation_space.shape[0], self.env.action_space.shape[0], nhid).to(self.device)
        self.net_crt = model.ModelCritic(self.env.observation_space.shape[0], nhid).to(self.device)

        self.opt_crt = optim.Adam(self.net_crt.parameters(), lr=lr_critic)

        self.batch = []

        self.gamma = gamma

    def loop(self, test_iters, target, maxeps):

        maxeps = np.inf if maxeps is None else maxeps

        best_reward = None
        tstart = time()

        rewards_steps = None

        evaluations = []

        for episode_idx, exp in enumerate(self.exp_source):

            rewards_steps = self.exp_source.pop_rewards_steps()

            tcurr = time()

            if episode_idx == maxeps:
                break
            
            if rewards_steps:
                rewards, steps = zip(*rewards_steps)

            if episode_idx % test_iters == 0:
                reward, steps = test_net(self.net_act, self.env, device=self.device)
                print('Episode %07d:\treward = %+.3f,\tsteps = %d' % (episode_idx, reward, steps))
                model_fname = self.models_path + ('%+.3f_%d.dat' % (reward, episode_idx))
                evaluations.append((episode_idx+1, reward))
                if best_reward is None or best_reward < reward:
                    if best_reward is not None:
                        print('\n* Best reward updated: %.3f -> %.3f *\n' % (best_reward, reward))
                        self._save(model_fname)
                    best_reward = reward
                if target is not None and reward >= target:
                    print('Target %f achieved; saving %s' % (target,model_fname))
                    self._save(model_fname)
                    break

            self.update(exp, maxeps)

        np.save(self.runs_path+('' if best_reward is None else ('%f'%best_reward)), evaluations)

    def _save(self, model_fname):

        d = self._clean(self.net_act.state_dict())

        torch.save((d,self.env_name,self.nhid), open(model_fname, 'wb'))

    def _clean(self, net):

        return net


def make_learn_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', default='Pendulum-v0', help='Environment id')
    parser.add_argument('--nhid', default=64, type=int, help='Hidden units')
    parser.add_argument('--maxeps', default=None, type=int, help='Maximum number of episodes')
    parser.add_argument('--target', type=float, default=np.inf, help='Quitting criterion for average reward')
    parser.add_argument('--gamma', default=0.99, help='Discount factor')
    parser.add_argument('--test-iters', default=100, type=float, help='How often to test and save best')

    parser.add_argument('--cuda', default=False, action='store_true', help='Enable CUDA')
    parser.add_argument('--datafile', required=False, help='Name of data file to load')
    return parser

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
    p2 = - torch.log(torch.sqrt(2 * np.pi * torch.exp(logstd_v)))
    return p1 + p2


