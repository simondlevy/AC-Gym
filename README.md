This repository combines code from three sources, q.v. for details:

* Chapter 19 of Pack Publishing's
[Deep-Reinforcement-Learning-Hands-On-Second-Edition](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition) 

* Scott Fujimoto's [TD3](https://www.youtube.com/watch?v=ZwU9SqO0udU)

* Max Lapan's [ptan](https://github.com/Shmuma/ptan)

My goal with this repository is to bring all these algorithms together in a single place, with a simple, uniform
command-line interface and minimal external dependencies.

## Quickstart

```python3 td3-learn.py --target -500```

This will run the [TD3](https://towardsdatascience.com/td3-learning-to-run-with-ai-40dfc512f93)
algorithm on the default environment
([Pendulum-v0](https://gym.openai.com/envs/Pendulum-v0/)) until an average
reward of  -500 is reached (which takes about 23 seconds on my Asus Predator
Helios laptop).  Once the program completes, you can display the results by
running

```python3 test.py models/td3-Pendulum-v0-<REWARD>.dat```

where ```<REWARD>``` is the reward value.
