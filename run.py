#!/usr/bin/python
#  -*- coding: utf-8 -*-
# author:  <yao62995@gmail.com> 

import os
import gym
import numpy as np

from DDPG_deep_deterministic_policy_gradient import DDPG

stage_train = 1e4
stage_test = 1e4
total_train = 1e6
t_max = 1e4


def run_episode(env, agent, test=True, monitor=False):
    env.monitor.configure(lambda _: test and monitor)
    state = env.reset()
    state = np.reshape(state, (1, env.observation_space.shape[0]))
    agent.explore_noise.reset()
    R = 0  # return
    t = 1
    term = False
    while not term:
        # env.render()
        action = agent.get_action(state, with_noise=not test)
        state_n, reward, term, info = env.step(action)
        state_n = np.reshape(state_n, (1, env.observation_space.shape[0]))
        term = (t >= t_max) or term
        if not test:
            agent.feedback(state, action, reward, term, state_n)
        state = state_n
        R += reward
        t += 1
    return R, t


if __name__ == "__main__":
    # experiment = "InvertedPendulum-v1"
    experiment = "Ant-v1"
    env = gym.make(experiment)
    save_dir = './result/%s/monitor/' % experiment
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    env.monitor.start(save_dir, video_callable=lambda _: False, force=True)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_range = (env.action_space.low, env.action_space.high)
    agent = DDPG(state_dim, action_dim, action_range=action_range, frame_seq_num=1)
    t_train, t_test = 0, 0
    while t_train < total_train:
        # test
        T = t_test
        R = []
        # env.monitor.start(save_dir, video_callable=lambda _: False, resume=True)
        while t_test - T < stage_test:
            r, t = run_episode(env, agent, test=True, monitor=(len(R) == 0))
            R.append(r)
            t_test += t
        avr = sum(R) / len(R)
        print('Average test return\t{} after {} timesteps of training'.format(avr, t_train))
        # env.monitor.close()
        # train
        T = t_train
        R = []
        while t_train - T < stage_train:
            r, t = run_episode(env, agent, test=False)
            R.append(r)
            t_train += t
        avr = sum(R) / len(R)
        print('Average train return\t{} after {} timesteps of training'.format(avr, t_train))

    env.monitor.close()
