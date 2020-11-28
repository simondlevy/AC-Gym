#!/usr/bin/env python3
import gym

from libs import Solver, ptan, model, common, make_learn_parser, parse_args, make_nets

import torch.optim as optim
import torch.nn.functional as F

class SAC(Solver):

    def __init__(self, args, device, net_act, net_crt, env, exp_source):

        Solver.__init__(self, args, device, net_act, net_crt)

        self.exp_source = exp_source

        self.twinq_net = model.ModelSACTwinQ( env.observation_space.shape[0], env.action_space.shape[0]).to(device)

        self.tgt_net_crt = ptan.agent.TargetNet(net_crt)

        self.buffer = ptan.experience.ExperienceReplayBuffer(self.exp_source, buffer_size=args.replay_size)
        self.act_opt = optim.Adam(net_act.parameters(), lr=args.lr_actor)
        self.crt_opt = optim.Adam(net_crt.parameters(), lr=args.lr_values)
        self.twinq_opt = optim.Adam(self.twinq_net.parameters(), lr=args.lr_values)

    def update(self, exp, maxeps):

        self.buffer.populate(1)
        rewards_steps = self.exp_source.pop_rewards_steps()
        if rewards_steps:
            rewards, steps = zip(*rewards_steps)

        if len(self.buffer) < self.args.replay_initial:
            return

        batch = self.buffer.sample(self.args.batch_size)
        states_v, actions_v, ref_vals_v, ref_q_v = \
            common.unpack_batch_sac(
                batch, self.tgt_net_crt.target_model,
                self.twinq_net, self.net_act, self.args.gamma,
                self.args.entropy_alpha, self.device)

        # train TwinQ
        self.twinq_opt.zero_grad()
        q1_v, q2_v = self.twinq_net(states_v, actions_v)
        q1_loss_v = F.mse_loss(q1_v.squeeze(),
                               ref_q_v.detach())
        q2_loss_v = F.mse_loss(q2_v.squeeze(),
                               ref_q_v.detach())
        q_loss_v = q1_loss_v + q2_loss_v
        q_loss_v.backward()
        self.twinq_opt.step()

        # Critic
        self.crt_opt.zero_grad()
        val_v = self.net_crt(states_v)
        v_loss_v = F.mse_loss(val_v.squeeze(), ref_vals_v.detach())
        v_loss_v.backward()
        self.crt_opt.step()

        # Actor
        self.act_opt.zero_grad()
        acts_v = self.net_act(states_v)
        q_out_v, _ = self.twinq_net(states_v, acts_v)
        act_loss = -q_out_v.mean()
        act_loss.backward()
        self.act_opt.step()

        self.tgt_net_crt.alpha_sync(alpha=1 - 1e-3)

    def clean(self, net):

        return net

def main():

    parser = make_learn_parser()

    parser.add_argument('--batch-size', default=64, type=int, help='Batch size')
    parser.add_argument('--lr-actor', default=1e-4, type=float, help='Learning rate for actor')
    parser.add_argument('--lr-values', default=1e-4, type=float, help='Learning rate for values')
    parser.add_argument('--replay-size', default=100000, type=int, help='Replay size')
    parser.add_argument('--replay-initial', default=10000, type=int, help='Initial replay size')
    parser.add_argument('--entropy-alpha', default=0.1, type=float, help='Entropy alpha')

    args, device, models_path, runs_path, test_env, maxeps, maxsec = parse_args(parser, 'sac')

    env = gym.make(args.env)

    net_act, net_crt = make_nets(args, env, device)

    agent = model.AgentDDPG(net_act, device=device)

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=args.gamma, steps_count=1)

    solver = SAC(args, device, net_act, net_crt, env,  exp_source)

    solver.loop(args, exp_source, maxeps, maxsec, test_env, models_path, runs_path)

if __name__ == '__main__':
    main()
