#!/usr/bin/env python3
import math

from libs import ptan, model, common, kfac, calc_logprob, make_learn_parser, parse_args, loop

import gym
import torch
import torch.optim as optim
import torch.nn.functional as F

class ACKTR:

    def __init__(self, args, device, net_act, net_crt):

        self.args = args
        self.device = device
        self.batch = []
        self.net_act = net_act
        self.net_crt = net_crt

        self.opt_act = kfac.KFACOptimizer(net_act, lr=args.lr_actor)
        self.opt_crt = optim.Adam(net_crt.parameters(), lr=args.lr_critic)

    def update(self, exp, maxeps):

        self.batch.append(exp)
        if len(self.batch) < self.args.batch_size:
            return

        states_v, actions_v, vals_ref_v = \
            common.unpack_batch_a2c(self.batch, self.net_crt, 
                    last_val_gamma=self.args.gamma ** self.args.reward_steps, device=self.device)
        self.batch.clear()

        self.opt_crt.zero_grad()
        value_v = self.net_crt(states_v)
        loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)
        loss_value_v.backward()
        self.opt_crt.step()

        mu_v = net_act(states_v)
        log_prob_v = calc_logprob(mu_v, net_act.logstd, actions_v)
        if self.opt_act.steps % self.opt_act.Ts == 0:
            self.opt_act.zero_grad()
            pg_fisher_loss = -log_prob_v.mean()
            self.opt_act.acc_stats = True
            pg_fisher_loss.backward(retain_graph=True)
            self.opt_act.acc_stats = False

        self.opt_act.zero_grad()
        adv_v = vals_ref_v.unsqueeze(dim=-1) - value_v.detach()
        loss_policy_v = -(adv_v * log_prob_v).mean()
        entropy_loss_v = self.args.entropy_beta * (-(torch.log(2*math.pi*torch.exp(self.net_act.logstd)) + 1)/2).mean()
        loss_v = loss_policy_v + entropy_loss_v
        loss_v.backward()
        self.opt_act.step()

    def clean(old):

        # Correct for different key names in ACKTR
        new = {}
        for oldkey in old.keys():
            newkey = oldkey.replace('.module','').replace('add_bias._', '')
            new[newkey] = old[oldkey]
            if 'bias' in newkey:
                new[newkey] = new[newkey].flatten()
        return new

if __name__ == '__main__':

    parser = make_learn_parser()

    parser.add_argument('--reward-steps', default=5, type=int, help='Reward steps')
    parser.add_argument('--batch-size', default=32, type=int, help='Batch size')
    parser.add_argument('--lr-actor', default=1e-3, type=float, help='Learning rate for actor')
    parser.add_argument('--lr-critic', default=1e-3, type=float, help='Learning rate for critic')
    parser.add_argument('--entropy-beta', default=1e-3, type=float, help='Entropy beta')
    parser.add_argument('--envs-count', default=16, type=int, help='Environments count')

    args, device, models_path, runs_path, test_env, maxeps, maxsec = parse_args(parser, 'acktr')

    envs = [gym.make(args.env) for _ in range(args.envs_count)]

    net_act = model.ModelActor(envs[0].observation_space.shape[0], envs[0].action_space.shape[0], args.nhid).to(device)
    net_crt = model.ModelCritic(envs[0].observation_space.shape[0], args.nhid).to(device)

    agent = model.AgentA2C(net_act, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, args.gamma, steps_count=args.reward_steps)

    solver = ACKTR(args, device, net_act, net_crt)

    loop(args, exp_source, solver, maxeps, maxsec, test_env, models_path, runs_path)
