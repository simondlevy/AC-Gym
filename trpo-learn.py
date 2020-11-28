#!/usr/bin/env python3
import gym

from libs import ptan, model, trpo, calc_logprob, make_learn_parser, parse_args, make_nets, loop

import torch
import torch.optim as optim
import torch.nn.functional as F

class TRPO:

    def __init__(self, args, device, net_act, net_crt):

        self.args = args
        self.device = device
        self.batch = []
        self.net_act = net_act
        self.net_crt = net_crt

        self.opt_crt = optim.Adam(net_crt.parameters(), lr=args.lr)
        self.trajectory = []

    def update(self, exp, maxeps):

        self.trajectory.append(exp)
        if len(self.trajectory) < self.args.traj_size:
            return

        traj_states = [t[0].state for t in self.trajectory]
        traj_actions = [t[0].action for t in self.trajectory]
        traj_states_v = torch.FloatTensor(traj_states).to(self.device)
        traj_actions_v = torch.FloatTensor(traj_actions).to(self.device)
        traj_adv_v, traj_ref_v = self.calc_adv_ref(traj_states_v)
        mu_v = net_act(traj_states_v)
        old_logprob_v = calc_logprob(mu_v, self.net_act.logstd, traj_actions_v)

        # normalize advantages
        traj_adv_v = (traj_adv_v - torch.mean(traj_adv_v)) / torch.std(traj_adv_v)

        # drop last entry from the trajectory, an our adv and ref value calculated without it
        self.trajectory = self.trajectory[:-1]
        old_logprob_v = old_logprob_v[:-1].detach()
        traj_states_v = traj_states_v[:-1]
        traj_actions_v = traj_actions_v[:-1]

        # critic step
        self.opt_crt.zero_grad()
        value_v = self.net_crt(traj_states_v)
        loss_value_v = F.mse_loss(
            value_v.squeeze(-1), traj_ref_v)
        loss_value_v.backward()
        self.opt_crt.step()

        # actor step
        def get_loss():
            mu_v = net_act(traj_states_v)
            logprob_v = calc_logprob(mu_v, self.net_act.logstd, traj_actions_v)
            dp_v = torch.exp(logprob_v - old_logprob_v)
            action_loss_v = -traj_adv_v.unsqueeze(dim=-1)*dp_v
            return action_loss_v.mean()

        def get_kl():
            mu_v = self.net_act(traj_states_v)
            logstd_v = self.net_act.logstd
            mu0_v = mu_v.detach()
            logstd0_v = logstd_v.detach()
            std_v = torch.exp(logstd_v)
            std0_v = std_v.detach()
            v = (std0_v ** 2 + (mu0_v - mu_v) ** 2) / \
                (2.0 * std_v ** 2)
            kl = logstd_v - logstd0_v + v - 0.5
            return kl.sum(1, keepdim=True)

        trpo.trpo_step(self.net_act, get_loss, get_kl, self.args.maxkl, self.args.damping, device=self.device)

        self.trajectory.clear()

    def clean(self, net):

        return net

    def calc_adv_ref(self, states_v):
        '''
        By trajectory calculate advantage and 1-step ref value
        :param trajectory: list of Experience objects
        :param net_crt: critic network
        :return: tuple with advantage numpy array and reference values
        '''
        values_v = self.net_crt(states_v)
        values = values_v.squeeze().data.cpu().numpy()
        # generalized advantage estimator: smoothed version of the advantage
        last_gae = 0.0
        result_adv = []
        result_ref = []
        for val, next_val, (exp,) in zip(reversed(values[:-1]), reversed(values[1:]), reversed(self.trajectory[:-1])):
            if exp.done:
                delta = exp.reward - val
                last_gae = delta
            else:
                delta = exp.reward + self.args.gamma * next_val - val
                last_gae = delta + self.args.gamma * self.args.gae_lambda * last_gae
            result_adv.append(last_gae)
            result_ref.append(last_gae + val)

        adv_v = torch.FloatTensor(list(reversed(result_adv))).to(self.device)
        ref_v = torch.FloatTensor(list(reversed(result_ref))).to(self.device)
        return adv_v, ref_v

if __name__ == '__main__':

    parser = make_learn_parser()

    parser.add_argument('--lr', default=1e-3, type=float, help='Critic learning rate')
    parser.add_argument('--maxkl', default=0.01, type=float, help='Maximum KL divergence')
    parser.add_argument('--damping', default=0.1, type=float, help='Damping')
    parser.add_argument('--gae-lambda', default=0.95, type=float, help='Lambda for Generalized Advantage Estimation')
    parser.add_argument('--traj-size', default=2049, type=int, help='Trajectory size')

    args, device, models_path, runs_path, test_env, maxeps, maxsec = parse_args(parser, 'trpo')

    env = gym.make(args.env)

    net_act, net_crt = make_nets(args, env, device)

    agent = model.AgentA2C(net_act, device=device)
    exp_source = ptan.experience.ExperienceSource(env, agent, steps_count=1)

    solver = TRPO(args, device, net_act, net_crt)

    loop(args, exp_source, solver, maxeps, maxsec, test_env, models_path, runs_path)


