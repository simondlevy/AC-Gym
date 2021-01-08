#!/usr/bin/env python3
import math

from ac_gym import Solver, model, common, calc_logprob
from ac_gym import gym_make, make_learn_parser
from ac_gym.ptan.experience import ExperienceSourceFirstLast

import torch
import torch.optim as optim
import torch.nn.functional as F


class A2C(Solver):

    def __init__(self, args):

        envs = [gym_make(args.env) for _ in range(args.envs_count)]

        Solver.__init__(self, args, 'a2c')

        agent = model.AgentA2C(self.net_act, device=self.device)

        steps = args.reward_steps
        self.exp_source = ExperienceSourceFirstLast(envs,
                                                    agent,
                                                    args.gamma,
                                                    steps_count=steps)

        self.opt_act = optim.Adam(self.net_act.parameters(), lr=args.lr_actor)

        self.batch_size = args.batch_size
        self.reward_steps = args.reward_steps
        self.entropy_beta = args.entropy_beta

    def update(self, exp):

        self.batch.append(exp)

        if len(self.batch) < self.batch_size:
            return

        g = self.gamma ** self.reward_steps
        states_v, actions_v, vals_ref_v = (
            common.unpack_batch_a2c(self.batch,
                                    self.net_crt,
                                    last_val_gamma=g,
                                    device=self.device))
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
        entropy_loss_v = (self.entropy_beta *
                          (-(torch.log(2 *
                                       math.pi*torch.exp(self.net_act.logstd))
                             + 1)/2).mean())
        loss_v = loss_policy_v + entropy_loss_v
        loss_v.backward()
        self.opt_act.step()


def main():

    parser = make_learn_parser()

    parser.add_argument('--reward-steps', default=5, type=int,
                        help='Reward steps')
    parser.add_argument('--batch-size', default=32, type=int,
                        help='Batch size')
    parser.add_argument('--lr-actor', default=1e-5, type=float,
                        help='Learning rate for actor')
    parser.add_argument('--lr-critic', default=1e-3, type=float,
                        help='Learning rate for critic')
    parser.add_argument('--entropy-beta', default=1e-3, type=float,
                        help='Entropy beta')
    parser.add_argument('--envs-count', default=16, type=int,
                        help='Environments count')

    args = parser.parse_args()

    A2C(args).loop()


main()
