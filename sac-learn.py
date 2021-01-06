#!/usr/bin/env python3
import gym

from ac_gym import Solver, ptan, model, common, make_learn_parser
from ac_gym.ptan.experience import ExperienceSourceFirstLast
from ac_gym.ptan.experience import ExperienceReplayBuffer

import torch.optim as optim
import torch.nn.functional as F


class SAC(Solver):

    def __init__(self, args):

        env = gym.make(args.env)

        Solver.__init__(self, args, 'sac')

        agent = model.AgentDDPG(self.net_act, device=self.device)

        self.exp_source = ExperienceSourceFirstLast(env,
                                                    agent,
                                                    gamma=args.gamma,
                                                    steps_count=1)

        self.twinq_net = (model.ModelSACTwinQ(env.observation_space.shape[0],
                          env.action_space.shape[0]).to(self.device))

        self.tgt_net_crt = ptan.agent.TargetNet(self.net_crt)

        bs = args.replay_size
        self.buffer = ExperienceReplayBuffer(self.exp_source, buffer_size=bs)
        self.act_opt = optim.Adam(self.net_act.parameters(), lr=args.lr_actor)
        self.twinq_opt = optim.Adam(self.twinq_net.parameters(),
                                    lr=args.lr_critic)

        self.replay_size = args.replay_size
        self.replay_initial = args.replay_initial
        self.batch_size = args.batch_size
        self.entropy_alpha = args.entropy_alpha

    def update(self, exp):

        self.buffer.populate(1)
        rewards_steps = self.exp_source.pop_rewards_steps()
        if rewards_steps:
            rewards, steps = zip(*rewards_steps)

        if len(self.buffer) < self.replay_initial:
            return

        batch = self.buffer.sample(self.batch_size)
        states_v, actions_v, ref_vals_v, ref_q_v = \
            common.unpack_batch_sac(
                batch, self.tgt_net_crt.target_model,
                self.twinq_net, self.net_act, self.gamma,
                self.entropy_alpha, self.device)

        # train TwinQ
        self.twinq_opt.zero_grad()
        q1_v, q2_v = self.twinq_net(states_v, actions_v)
        q1_loss_v = F.mse_loss(q1_v.squeeze(), ref_q_v.detach())
        q2_loss_v = F.mse_loss(q2_v.squeeze(), ref_q_v.detach())
        q_loss_v = q1_loss_v + q2_loss_v
        q_loss_v.backward()
        self.twinq_opt.step()

        # Critic
        self.opt_crt.zero_grad()
        val_v = self.net_crt(states_v)
        v_loss_v = F.mse_loss(val_v.squeeze(), ref_vals_v.detach())
        v_loss_v.backward()
        self.opt_crt.step()

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

    parser.add_argument('--batch-size', default=64, type=int,
                        help='Batch size')
    parser.add_argument('--lr-actor', default=1e-4, type=float,
                        help='Learning rate for actor')
    parser.add_argument('--lr-critic', default=1e-4, type=float,
                        help='Learning rate for values')
    parser.add_argument('--replay-size', default=100000, type=int,
                        help='Replay size')
    parser.add_argument('--replay-initial', default=10000, type=int,
                        help='Initial replay size')
    parser.add_argument('--entropy-alpha', default=0.1, type=float,
                        help='Entropy alpha')

    args = parser.parse_args()

    SAC(args).loop()


main()
