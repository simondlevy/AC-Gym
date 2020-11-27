#!/usr/bin/env python3
import os
import time
import gym
from tensorboardX import SummaryWriter

from libs import ptan, model, common, test_net, make_learn_parser, parse_args, make_nets

import torch
import torch.optim as optim
import torch.nn.functional as F

GAMMA = 0.99
BATCH_SIZE = 64
LR_ACTS = 1e-4
LR_VALS = 1e-4
REPLAY_SIZE = 100000
REPLAY_INITIAL = 10000
SAC_ENTROPY_ALPHA = 0.1

if __name__ == '__main__':

    parser = make_learn_parser()

    args, device, save_path, test_env, maxeps, maxsec = parse_args(parser, 'sac')

    env = gym.make(args.env)

    net_act, net_crt = make_nets(args, env, device)

    twinq_net = model.ModelSACTwinQ( env.observation_space.shape[0], env.action_space.shape[0]).to(device)

    tgt_net_crt = ptan.agent.TargetNet(net_crt)

    writer = SummaryWriter(comment='-sac_' + args.env)
    agent = model.AgentDDPG(net_act, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=GAMMA, steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(
        exp_source, buffer_size=REPLAY_SIZE)
    act_opt = optim.Adam(net_act.parameters(), lr=LR_ACTS)
    crt_opt = optim.Adam(net_crt.parameters(), lr=LR_VALS)
    twinq_opt = optim.Adam(twinq_net.parameters(), lr=LR_VALS)

    step_idx = 0
    best_reward = None
    tstart = time.time()

    with ptan.common.utils.RewardTracker(writer) as tracker:

        with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:

            while True:

                if len(tracker.total_rewards) >= maxeps:
                    break

                step_idx += 1
                buffer.populate(1)
                rewards_steps = exp_source.pop_rewards_steps()
                if rewards_steps:
                    rewards, steps = zip(*rewards_steps)
                    tb_tracker.track('episode_steps', steps[0], step_idx)
                    tracker.reward(rewards[0], step_idx)

                if len(buffer) < REPLAY_INITIAL:
                    continue

                batch = buffer.sample(BATCH_SIZE)
                states_v, actions_v, ref_vals_v, ref_q_v = \
                    common.unpack_batch_sac(
                        batch, tgt_net_crt.target_model,
                        twinq_net, net_act, GAMMA,
                        SAC_ENTROPY_ALPHA, device)

                tb_tracker.track('ref_v', ref_vals_v.mean(), step_idx)
                tb_tracker.track('ref_q', ref_q_v.mean(), step_idx)

                # train TwinQ
                twinq_opt.zero_grad()
                q1_v, q2_v = twinq_net(states_v, actions_v)
                q1_loss_v = F.mse_loss(q1_v.squeeze(),
                                       ref_q_v.detach())
                q2_loss_v = F.mse_loss(q2_v.squeeze(),
                                       ref_q_v.detach())
                q_loss_v = q1_loss_v + q2_loss_v
                q_loss_v.backward()
                twinq_opt.step()
                tb_tracker.track('loss_q1', q1_loss_v, step_idx)
                tb_tracker.track('loss_q2', q2_loss_v, step_idx)

                # Critic
                crt_opt.zero_grad()
                val_v = net_crt(states_v)
                v_loss_v = F.mse_loss(val_v.squeeze(),
                                      ref_vals_v.detach())
                v_loss_v.backward()
                crt_opt.step()
                tb_tracker.track('loss_v', v_loss_v, step_idx)

                # Actor
                act_opt.zero_grad()
                acts_v = net_act(states_v)
                q_out_v, _ = twinq_net(states_v, acts_v)
                act_loss = -q_out_v.mean()
                act_loss.backward()
                act_opt.step()
                tb_tracker.track('loss_act', act_loss, step_idx)

                tgt_net_crt.alpha_sync(alpha=1 - 1e-3)

                tcurr = time.time()

                if (tcurr-tstart) >= maxsec:
                    break

                if step_idx % args.test_iters == 0:
                    reward, steps = test_net(net_act, test_env, device=device)
                    print('Test done in %.2f sec, reward %.3f, steps %d' % (time.time() - tcurr, reward, steps))
                    writer.add_scalar('test_reward', reward, step_idx)
                    writer.add_scalar('test_steps', steps, step_idx)
                    name = '%+.3f_%d.dat' % (reward, step_idx)
                    fname = save_path + name
                    if best_reward is None or best_reward < reward:
                        if best_reward is not None:
                            print('Best reward updated: %.3f -> %.3f' % (best_reward, reward))
                            torch.save(net_act.state_dict(), fname)
                        best_reward = reward
                    if args.target is not None and reward >= args.target:
                        print('Target %f achieved; saving %s' % (args.target,fname))
                        torch.save(net_act.state_dict(), fname)
                        break

    pass
