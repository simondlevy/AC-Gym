#!/usr/bin/env python3
import os
import math
import time
from tensorboardX import SummaryWriter

from libs import ptan, model, common, kfac, test_net, calc_logprob, make_learn_parser, parse_args

import gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

def clean_act_net(d):
    # Correct for different key names in ACKTR
    newd = {}
    for key in d.keys():
        newd[key.replace('.module','').replace('add_bias._', '')] = d[key]
    return newd

if __name__ == '__main__':

    parser = make_learn_parser()

    parser.add_argument('--reward-steps', default=5, type=int, help='Reward steps')
    parser.add_argument('--batch-size', default=32, type=int, help='Batch size')
    parser.add_argument('--lr-actor', default=1e-3, type=float, help='Learning rate for actor')
    parser.add_argument('--lr-critic', default=1e-3, type=float, help='Learning rate for critic')
    parser.add_argument('--entropy-beta', default=1e-3, type=float, help='Entropy beta')
    parser.add_argument('--envs-count', default=16, type=int, help='Environments count')

    args, device, save_path, test_env, maxeps, maxsec = parse_args(parser, 'acktr')

    envs = [gym.make(args.env) for _ in range(args.envs_count)]

    net_act = model.ModelActor(envs[0].observation_space.shape[0], envs[0].action_space.shape[0], args.nhid).to(device)
    net_crt = model.ModelCritic(envs[0].observation_space.shape[0], args.nhid).to(device)

    writer = SummaryWriter(comment='-acktr_' + args.env)
    agent = model.AgentA2C(net_act, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, args.gamma, steps_count=args.reward_steps)

    opt_act = kfac.KFACOptimizer(net_act, lr=args.lr_actor)
    opt_crt = optim.Adam(net_crt.parameters(), lr=args.lr_critic)

    batch = []
    best_reward = None
    tstart = time.time()

    with ptan.common.utils.RewardTracker(writer) as tracker:

        with ptan.common.utils.TBMeanTracker(writer, batch_size=100) as tb_tracker:

            for step_idx, exp in enumerate(exp_source):

                if len(tracker.total_rewards) >= maxeps:
                    break

                rewards_steps = exp_source.pop_rewards_steps()

                if rewards_steps:
                    rewards, steps = zip(*rewards_steps)
                    tb_tracker.track('episode_steps', np.mean(steps), step_idx)
                    tracker.reward(np.mean(rewards), step_idx)

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

                batch.append(exp)
                if len(batch) < args.batch_size:
                    continue

                states_v, actions_v, vals_ref_v = \
                    common.unpack_batch_a2c(batch, net_crt, last_val_gamma=args.gamma ** args.reward_steps, device=device)
                batch.clear()

                opt_crt.zero_grad()
                value_v = net_crt(states_v)
                loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)
                loss_value_v.backward()
                opt_crt.step()

                mu_v = net_act(states_v)
                log_prob_v = calc_logprob(mu_v, net_act.logstd, actions_v)
                if opt_act.steps % opt_act.Ts == 0:
                    opt_act.zero_grad()
                    pg_fisher_loss = -log_prob_v.mean()
                    opt_act.acc_stats = True
                    pg_fisher_loss.backward(retain_graph=True)
                    opt_act.acc_stats = False

                opt_act.zero_grad()
                adv_v = vals_ref_v.unsqueeze(dim=-1) - value_v.detach()
                loss_policy_v = -(adv_v * log_prob_v).mean()
                entropy_loss_v = args.entropy_beta * (-(torch.log(2*math.pi*torch.exp(net_act.logstd)) + 1)/2).mean()
                loss_v = loss_policy_v + entropy_loss_v
                loss_v.backward()
                opt_act.step()

                tb_tracker.track('advantage', adv_v, step_idx)
                tb_tracker.track('values', value_v, step_idx)
                tb_tracker.track('batch_rewards', vals_ref_v, step_idx)
                tb_tracker.track('loss_entropy', entropy_loss_v, step_idx)
                tb_tracker.track('loss_policy', loss_policy_v, step_idx)
                tb_tracker.track('loss_value', loss_value_v, step_idx)
                tb_tracker.track('loss_total', loss_v, step_idx)
