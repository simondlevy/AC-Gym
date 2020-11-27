#!/usr/bin/env python3
import os
import time
import gym
from tensorboardX import SummaryWriter

from libs import ptan, model, test_net, calc_logprob, make_learn_parser, parse_args, make_nets

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

def calc_adv_ref(trajectory, net_crt, states_v, device='cpu'):
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
                                     reversed(trajectory[:-1])):
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
    return adv_v.to(device), ref_v.to(device)


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

    writer = SummaryWriter(comment='-ppo_' + args.env)
    agent = model.AgentA2C(net_act, device=device)
    exp_source = ptan.experience.ExperienceSource(env, agent, steps_count=1)

    opt_act = optim.Adam(net_act.parameters(), lr=args.lr_actor)
    opt_crt = optim.Adam(net_crt.parameters(), lr=args.lr_critic)

    trajectory = []
    best_reward = None
    tstart = time.time()

    with ptan.common.utils.RewardTracker(writer) as tracker:

        for step_idx, exp in enumerate(exp_source):

            if len(tracker.total_rewards) >= maxeps:
                break

            rewards_steps = exp_source.pop_rewards_steps()

            if rewards_steps:
                rewards, steps = zip(*rewards_steps)
                writer.add_scalar('episode_steps', np.mean(steps), step_idx)
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

            trajectory.append(exp)
            if len(trajectory) < args.traj_size:
                continue

            traj_states = [t[0].state for t in trajectory]
            traj_actions = [t[0].action for t in trajectory]
            traj_states_v = torch.FloatTensor(traj_states)
            traj_states_v = traj_states_v.to(device)
            traj_actions_v = torch.FloatTensor(traj_actions)
            traj_actions_v = traj_actions_v.to(device)
            traj_adv_v, traj_ref_v = calc_adv_ref(trajectory, net_crt, traj_states_v, device=device)
            mu_v = net_act(traj_states_v)
            old_logprob_v = calc_logprob( mu_v, net_act.logstd, traj_actions_v)

            # normalize advantages
            traj_adv_v = traj_adv_v - torch.mean(traj_adv_v)
            traj_adv_v /= torch.std(traj_adv_v)

            # drop last entry from the trajectory, an our adv and ref value calculated without it
            trajectory = trajectory[:-1]
            old_logprob_v = old_logprob_v[:-1].detach()

            sum_loss_value = 0.0
            sum_loss_policy = 0.0
            count_steps = 0

            for epoch in range(args.epochs):

                for batch_ofs in range(0, len(trajectory), args.batch_size):

                    batch_l = batch_ofs + args.batch_size
                    states_v = traj_states_v[batch_ofs:batch_l]
                    actions_v = traj_actions_v[batch_ofs:batch_l]
                    batch_adv_v = traj_adv_v[batch_ofs:batch_l]
                    batch_adv_v = batch_adv_v.unsqueeze(-1)
                    batch_ref_v = traj_ref_v[batch_ofs:batch_l]
                    batch_old_logprob_v = \
                        old_logprob_v[batch_ofs:batch_l]

                    # critic training
                    opt_crt.zero_grad()
                    value_v = net_crt(states_v)
                    loss_value_v = F.mse_loss(
                        value_v.squeeze(-1), batch_ref_v)
                    loss_value_v.backward()
                    opt_crt.step()

                    # actor training
                    opt_act.zero_grad()
                    mu_v = net_act(states_v)
                    logprob_pi_v = calc_logprob(
                        mu_v, net_act.logstd, actions_v)
                    ratio_v = torch.exp(
                        logprob_pi_v - batch_old_logprob_v)
                    surr_obj_v = batch_adv_v * ratio_v
                    c_ratio_v = torch.clamp(ratio_v, 1.0 - args.epsilon, 1.0 + args.epsilon)
                    clipped_surr_v = batch_adv_v * c_ratio_v
                    loss_policy_v = -torch.min(
                        surr_obj_v, clipped_surr_v).mean()
                    loss_policy_v.backward()
                    opt_act.step()

                    sum_loss_value += loss_value_v.item()
                    sum_loss_policy += loss_policy_v.item()
                    count_steps += 1

            trajectory.clear()
            writer.add_scalar('advantage', traj_adv_v.mean().item(), step_idx)
            writer.add_scalar('values', traj_ref_v.mean().item(), step_idx)
            writer.add_scalar('loss_policy', sum_loss_policy / count_steps, step_idx)
            writer.add_scalar('loss_value', sum_loss_value / count_steps, step_idx)

