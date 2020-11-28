#!/usr/bin/env python3
import math
import gym

from libs import Solver, ptan, model, common, calc_logprob, make_learn_parser, parse_args, make_nets, loop

import torch
import torch.optim as optim
import torch.nn.functional as F

class A2C(Solver):

    def __init__(self, args, device, net_act, net_crt):

        Solver.__init__(self, args, device, net_act, net_crt)

        self.opt_act = optim.Adam(net_act.parameters(), lr=args.lr_actor)
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

        self.opt_act.zero_grad()
        mu_v = self.net_act(states_v)
        adv_v = vals_ref_v.unsqueeze(dim=-1) - value_v.detach()
        log_prob_v = adv_v * calc_logprob(mu_v, self.net_act.logstd, actions_v)
        loss_policy_v = -log_prob_v.mean()
        entropy_loss_v = self.args.entropy_beta * (-(torch.log(2*math.pi*torch.exp(self.net_act.logstd)) + 1)/2).mean()
        loss_v = loss_policy_v + entropy_loss_v
        loss_v.backward()
        self.opt_act.step()

    def clean(self, net):

        return net

def main():

    parser = make_learn_parser()

    parser.add_argument('--reward-steps', default=5, type=int, help='Reward steps')
    parser.add_argument('--batch-size', default=32, type=int, help='Batch size')
    parser.add_argument('--lr-actor', default=1e-5, type=float, help='Learning rate for actor')
    parser.add_argument('--lr-critic', default=1e-3, type=float, help='Learning rate for critic')
    parser.add_argument('--entropy-beta', default=1e-3, type=float, help='Entropy beta')
    parser.add_argument('--envs-count', default=16, type=int, help='Environments count')

    args, device, models_path, runs_path, test_env, maxeps, maxsec = parse_args(parser, 'a2c')

    envs = [gym.make(args.env) for _ in range(args.envs_count)]

    net_act, net_crt = make_nets(args, envs[0], device)

    agent = model.AgentA2C(net_act, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, args.gamma, steps_count=args.reward_steps)

    solver = A2C(args, device, net_act, net_crt)

    loop(args, exp_source, solver, maxeps, maxsec, test_env, models_path, runs_path)

if __name__ == '__main__':

    main()

