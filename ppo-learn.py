#!usr/bin/env python3
import gym

from libs import ptan, model, calc_logprob, make_learn_parser, parse_args, make_nets, loop

import torch
import torch.optim as optim
import torch.nn.functional as F

class PPO:

    def __init__(self, args, device, net_act, net_crt):

        self.args = args
        self.device = device
        self.batch = []
        self.net_act = net_act
        self.net_crt = net_crt

        self.opt_act = optim.Adam(net_act.parameters(), lr=args.lr_actor)
        self.opt_crt = optim.Adam(net_crt.parameters(), lr=args.lr_critic)

        self.trajectory = []

    def update(self, exp, maxeps):

        self.trajectory.append(exp)
        if len(self.trajectory) < self.args.traj_size:
            return

        traj_states = [t[0].state for t in self.trajectory]
        traj_actions = [t[0].action for t in self.trajectory]
        traj_states_v = torch.FloatTensor(traj_states)
        traj_states_v = traj_states_v.to(self.device)
        traj_actions_v = torch.FloatTensor(traj_actions)
        traj_actions_v = traj_actions_v.to(self.device)
        traj_adv_v, traj_ref_v = self.calc_adv_ref(traj_states_v)
        mu_v = net_act(traj_states_v)
        old_logprob_v = calc_logprob( mu_v, self.net_act.logstd, traj_actions_v)

        # normalize advantages
        traj_adv_v = traj_adv_v - torch.mean(traj_adv_v)
        traj_adv_v /= torch.std(traj_adv_v)

        # drop last entry from the trajectory, an our adv and ref value calculated without it
        self.trajectory = self.trajectory[:-1]
        old_logprob_v = old_logprob_v[:-1].detach()

        sum_loss_value = 0.0
        sum_loss_policy = 0.0
        count_steps = 0

        for epoch in range(self.args.epochs):

            for batch_ofs in range(0, len(self.trajectory), self.args.batch_size):

                batch_l = batch_ofs + self.args.batch_size
                states_v = traj_states_v[batch_ofs:batch_l]
                actions_v = traj_actions_v[batch_ofs:batch_l]
                batch_adv_v = traj_adv_v[batch_ofs:batch_l]
                batch_adv_v = batch_adv_v.unsqueeze(-1)
                batch_ref_v = traj_ref_v[batch_ofs:batch_l]
                batch_old_logprob_v = \
                    old_logprob_v[batch_ofs:batch_l]

                # critic training
                self.opt_crt.zero_grad()
                value_v = self.net_crt(states_v)
                loss_value_v = F.mse_loss(
                    value_v.squeeze(-1), batch_ref_v)
                loss_value_v.backward()
                self.opt_crt.step()

                # actor training
                self.opt_act.zero_grad()
                mu_v = net_act(states_v)
                logprob_pi_v = calc_logprob(
                    mu_v, self.net_act.logstd, actions_v)
                ratio_v = torch.exp(
                    logprob_pi_v - batch_old_logprob_v)
                surr_obj_v = batch_adv_v * ratio_v
                c_ratio_v = torch.clamp(ratio_v, 1.0 - self.args.epsilon, 1.0 + self.args.epsilon)
                clipped_surr_v = batch_adv_v * c_ratio_v
                loss_policy_v = -torch.min(
                    surr_obj_v, clipped_surr_v).mean()
                loss_policy_v.backward()
                self.opt_act.step()

                sum_loss_value += loss_value_v.item()
                sum_loss_policy += loss_policy_v.item()
                count_steps += 1

        self.trajectory.clear()

    def clean(self, net):

        return net

    def calc_adv_ref(self, states_v):
        '''
        By trajectory calculate advantage and 1-step ref value
        :param trajectory: trajectory list
        :param net_crt: critic network
        :param states_v: states tensor
        :return: tuple with advantage numpy array and reference values
        '''
        values_v = net_crt(states_v)
        values = values_v.squeeze().data.cpu().numpy()
        # generalized advantage estimator: smoothed version of the advantage
        last_gae = 0.0
        result_adv = []
        result_ref = []
        for val, next_val, (exp,) in zip(reversed(values[:-1]),
                                         reversed(values[1:]),
                                         reversed(self.trajectory[:-1])):
            if exp.done:
                delta = exp.reward - val
                last_gae = delta
            else:
                delta = exp.reward + args.gamma * next_val - val
                last_gae = delta + args.gamma * args.gae_lambda * last_gae
            result_adv.append(last_gae)
            result_ref.append(last_gae + val)

        adv_v = torch.FloatTensor(list(reversed(result_adv)))
        ref_v = torch.FloatTensor(list(reversed(result_ref)))
        return adv_v.to(self.device), ref_v.to(device)


if __name__ == '__main__':

    parser = make_learn_parser()

    parser.add_argument('--gae-lambda', default=0.95, type=float, help='Lambda for Generalized Advantage Estimation')
    parser.add_argument('--traj-size', default=2049, type=int, help='Trajectory size')
    parser.add_argument('--lr-actor', default=1e-5, type=float, help='Learning rate for actor')
    parser.add_argument('--lr-critic', default=1e-4, type=float, help='Learning rate for critic')
    parser.add_argument('--epsilon', default=0.2, type=float, help='Clipping')
    parser.add_argument('--epochs', default=10, type=int, help='Epochs')
    parser.add_argument('--batch-size', default=64, type=int, help='Batch size')

    args, device, save_path, test_env, maxeps, maxsec = parse_args(parser, 'ppo')

    env = gym.make(args.env)

    net_act, net_crt = make_nets(args, env, device)

    agent = model.AgentA2C(net_act, device=device)
    exp_source = ptan.experience.ExperienceSource(env, agent, steps_count=1)

    solver = PPO(args, device, net_act, net_crt)

    loop(args, exp_source, solver, maxeps, maxsec, test_env, save_path)
