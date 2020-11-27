import numpy as np
import gym
import time

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed=None, eval_episodes=10, render=False):

    eval_env = gym.make(env_name)

    if seed is not None:
        eval_env.seed(seed + 100)

    total_reward = 0.

    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            if render:
                eval_env.render()
                time.sleep(.02)
            total_reward += reward

    return total_reward / eval_episodes
