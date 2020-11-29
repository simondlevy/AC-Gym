#!/usr/bin/env python3
import math

from libs import Solver, ptan, model, common, kfac, calc_logprob, make_learn_parser

import gym
import torch
import torch.nn.functional as F

class ACKTR(Solver):

    def __init__(self,
            env_name, 
            nhid,
            cuda,
            envs_count, 
            gamma, 
            reward_steps, 
            lr_actor, 
            lr_critic, 
            batch_size, 
            entropy_beta):

        envs = [gym.make(env_name) for _ in range(envs_count)]

        Solver.__init__(self, env_name, 'acktr', nhid, cuda, gamma, lr_critic)

        agent = model.AgentA2C(self.net_act, device=self.device)

        self. exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, gamma, steps_count=reward_steps)

        self.opt_act = kfac.KFACOptimizer(self.net_act, lr=lr_actor)

        self.batch_size = batch_size
        self.reward_steps = reward_steps
        self.entropy_beta = entropy_beta


    def update(self, exp, maxeps):

        self.batch.append(exp)
        if len(self.batch) < self.batch_size:
            return

        states_v, actions_v, vals_ref_v = \
            common.unpack_batch_a2c(self.batch, self.net_crt, 
                    last_val_gamma=self.gamma ** self.reward_steps, device=self.device)
        self.batch.clear()

        self.opt_crt.zero_grad()
        value_v = self.net_crt(states_v)
        loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)
        loss_value_v.backward()
        self.opt_crt.step()

        mu_v = self.net_act(states_v)
        log_prob_v = calc_logprob(mu_v, self.net_act.logstd, actions_v)
        if self.opt_act.steps % self.opt_act.Ts == 0:
            self.opt_act.zero_grad()
            pg_fisher_loss = -log_prob_v.mean()
            self.opt_act.acc_stats = True
            pg_fisher_loss.backward(retain_graph=True)
            self.opt_act.acc_stats = False

        self.opt_act.zero_grad()
        adv_v = vals_ref_v.unsqueeze(dim=-1) - value_v.detach()
        loss_policy_v = -(adv_v * log_prob_v).mean()
        entropy_loss_v = self.entropy_beta * (-(torch.log(2*math.pi*torch.exp(self.net_act.logstd)) + 1)/2).mean()
        loss_v = loss_policy_v + entropy_loss_v
        loss_v.backward()
        self.opt_act.step()

    def _clean(self, old):

        # Correct for different key names in ACKTR
        new = {}
        for oldkey in old.keys():
            newkey = oldkey.replace('.module','').replace('add_bias._', '')
            new[newkey] = old[oldkey]
            if 'bias' in newkey:
                new[newkey] = new[newkey].flatten()
        return new

def main():

    parser = make_learn_parser()

    parser.add_argument('--reward-steps', default=5, type=int, help='Reward steps')
    parser.add_argument('--batch-size', default=32, type=int, help='Batch size')
    parser.add_argument('--lr-actor', default=1e-3, type=float, help='Learning rate for actor')
    parser.add_argument('--lr-critic', default=1e-3, type=float, help='Learning rate for critic')
    parser.add_argument('--entropy-beta', default=1e-3, type=float, help='Entropy beta')
    parser.add_argument('--envs-count', default=16, type=int, help='Environments count')

    args = parser.parse_args()

    solver = ACKTR(
            args.env,
            args.nhid,
            args.cuda,
            args.envs_count, 
            args.gamma, 
            args.reward_steps, 
            args.lr_actor, 
            args.lr_critic, 
            args.batch_size, 
            args.entropy_beta)

    solver.loop(args.test_iters, args.target, args.maxeps)

if __name__ == '__main__':
    main()
