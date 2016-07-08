#!/usr/bin/python
#  -*- coding: utf-8 -*-
# author: yao62995@gmail.com

import random
from collections import deque
import numpy as np

from common import *


class DQN(Base):
    """
        Deep Q-learning Model
        ref:
            paper "Playing Atari with Deep Reinforcement Learning"
    """

    def __init__(self, state_dim, action_dim, train_dir="./dqn_models/", batch_size=32, learn_rate=1e-4,
                 observe=1e3, explore=1e6, replay_memory=5e4, gamma=0.99, init_epsilon=1.0, final_epsilon=0.05,
                 update_frequency=4, action_repeat=4, frame_seq_num=4):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # DQN parameters
        self.observe = observe
        self.explore = explore
        self.replay_memory = replay_memory
        self.gamma = gamma
        self.init_epsilon = init_epsilon
        self.final_epsilon = final_epsilon

        self.update_frequency = update_frequency
        self.action_repeat = action_repeat

        self.frame_seq_num = frame_seq_num
        self.train_dir = train_dir
        self.memory = deque()

        self.batch_size = batch_size
        self.learn_rate = learn_rate
        self.sess = None
        self.saver = None
        self.global_step = None
        self.ops = dict()
        # build graph
        self.build_graph()
        # init train parameters
        self.epsilon = self.init_epsilon

    def inference(self, _input):
        # first conv1
        conv1 = conv2d(_input, (8, 8, self.frame_seq_num, 32), "conv_1", stride=2)
        pool1 = max_pool(conv1, ksize=2, stride=2)
        norm1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm_1')
        # second layer: conv2
        conv_n = conv2d(norm1, (3, 3, 32, 32), "conv_2", stride=2)
        pool_n = max_pool(conv_n, ksize=2, stride=2)
        norm_n = tf.nn.lrn(pool_n, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm_n')

        reshape = tf.reshape(norm_n, [-1, 25 * 32])
        # dim = reshape.get_shape()[1].value
        fc1 = full_connect(reshape, (25 * 32, 256), "fc_1")
        logits = full_connect(fc1, (256, self.action_dim), "fc_2", activate=None)
        return logits

    def build_graph(self):
        with tf.Graph().as_default(), tf.device('/gpu:%d' % self.gpu_id):
            self.global_step = tf.get_variable('global_step', [],
                                               initializer=tf.constant_initializer(0), trainable=False)
            # init placeholder
            state_ph = tf.placeholder(tf.float32, [None, 80, 80, self.frame_seq_num])
            action_ph = tf.placeholder(tf.float32, shape=[None, self.action_dim])
            target_ph = tf.placeholder(tf.float32, shape=[None])
            # init model
            logits = self.inference(state_ph)
            # calculate loss
            predict_act = tf.reduce_sum(tf.mul(logits, action_ph), reduction_indices=1)
            p_loss = tf.reduce_mean(tf.square(target_ph - predict_act))
            l2_loss = tf.add_n(tf.get_collection('losses'), name='l2_loss')
            total_loss = p_loss + l2_loss
            # optimizer
            optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(total_loss, global_step=self.global_step)
            self.saver = tf.train.Saver(max_to_keep=5)
            self.sess = tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False)
            )
            self.sess.run(tf.initialize_all_variables())
            # restore model
            restore_model(self.sess, self.train_dir, self.saver)
            # define ops
            self.ops["logits"] = lambda obs: self.sess.run([optimizer], feed_dict={state_ph: obs})
            self.ops["train_q"] = lambda obs, act, q_t: self.sess.run([optimizer, total_loss, self.global_step],
                                                                      feed_dict={state_ph: obs, action_ph: act,
                                                                                 target_ph: q_t})

    def get_action(self, state):
        if random.random() <= self.epsilon:  # random select
            action = random.randint(0, self.actions - 1)
        else:
            action = np.argmax(self.ops["logits"]([state])[0][0])
        return action

    def feedback(self, state, action, reward, terminal, state_n):
        self.time_step += 1
        # scale down epsilon
        if self.time_step > self.observe and self.epsilon > self.final_epsilon:
            self.epsilon -= (self.init_epsilon - self.final_epsilon) / self.explore
        # save replay memory
        self.replay_memory.append((state, action, reward, terminal, state_n))
        if len(self.replay_memory) > self.replay_memory_size:
            self.replay_memory.popleft()
        if self.time_step > self.observe and self.time_step % self.update_frequency == 0:
            for _ in xrange(self.train_repeat):
                # train mini-batch from replay memory
                mini_batch = random.sample(self.replay_memory, self.batch_size)
                batch_state, batch_action = [], []
                batch_target_q = []
                for batch_i, sample in enumerate(mini_batch):
                    b_state, b_action, b_reward, b_terminal, b_state_n = sample
                    if b_terminal:
                        target_q = b_reward
                    else:  # compute target q values
                        target_q = b_reward + self.gamma * np.max(self.ops["logits_target"]([b_state_n]))
                    batch_state.append(b_state)
                    batch_action.append(b_action)
                    batch_target_q.append(target_q)
                # update actor network (theta_p)
                _, p_loss = self.ops["train_p"](batch_state)
                # update critic network (theta_q)
                _, global_step, q_loss = self.ops["train_q"](batch_state, batch_action, batch_target_q)
                if self.time_step % 1e3 == 0:
                    logger.info("step=%d, p_loss=%.6f, q_loss=%.6f" % (global_step, p_loss, q_loss))
        if self.time_step % self.update_target_freq == 0:
            self.update_target_network()
        if self.time_step % 3e4 == 0:
            save_model(self.sess, self.train_dir, self.saver, "ddqn-", global_step=self.global_step)

    # def train(self):
    #     # training
    #     max_reward = 0
    #     epoch = 0
    #     train_step = 0
    #     state_desc = "observe"
    #     epsilon = self.init_epsilon
    #     env = Environment()
    #     state_ph, action_ph, target_ph, optimizer, logits, total_loss, global_step = self.graph_ops
    #     while True:  # loop epochs
    #         epoch += 1
    #         # initial state
    #         env.reset_game()
    #         # initial state sequences
    #         state_seq = np.empty((80, 80, self.frame_seq_num))
    #         for i in range(self.frame_seq_num):
    #             state = env.get_state()
    #             state_seq[:, :, i] = state
    #         stage_reward = 0
    #         while True:  # loop game frames
    #             # select action by ε-greedy policy
    #             if random.random() <= epsilon:  # random select
    #                 action = random.randint(0, self.actions - 1)
    #             else:
    #                 action = np.argmax(self.net.predict([state_seq])[0])
    #             # carry out selected action
    #             state_n, reward_n, terminal_n = env.step_forward(action)
    #             state_n = np.reshape(state_n, (80, 80, 1))
    #             state_seq_n = np.append(state_seq[:, :, : (self.frame_seq_num - 1)], state_n, axis=2)
    #             # scale down epsilon
    #             if train_step > self.observe and epsilon > self.final_epsilon:
    #                 epsilon -= (self.init_epsilon - self.final_epsilon) / self.explore
    #             # store experience
    #             act_onehot = np.zeros(self.actions)
    #             act_onehot[action] = 1
    #             self.memory.append((state_seq, act_onehot, reward_n, state_seq_n, terminal_n))
    #             if len(self.memory) > self.replay_memory:
    #                 self.memory.popleft()
    #             # minibatch train
    #             if len(self.memory) > self.observe and train_step % self.update_frequency == 0:
    #                 for _ in xrange(self.action_repeat):
    #                     mini_batch = random.sample(self.memory, self.batch_size)
    #                     batch_state_seq = [item[0] for item in mini_batch]
    #                     batch_action = [item[1] for item in mini_batch]
    #                     batch_reward = [item[2] for item in mini_batch]
    #                     batch_state_seq_n = [item[3] for item in mini_batch]
    #                     batch_terminal = [item[4] for item in mini_batch]
    #                     # predict
    #                     target_rewards = []
    #                     batch_pred_act_n = self.sess.run([logits], feed_dict={state_ph: batch_state_seq_n})
    #                     for i in xrange(len(mini_batch)):
    #                         if batch_terminal[i]:
    #                             t_r = batch_reward[i]
    #                         else:
    #                             t_r = batch_reward[i] + self.gamma * np.max(batch_pred_act_n[i])
    #                         target_rewards.append(t_r)
    #                     # train Q network
    #                     _, loss, global_step_val = self.sess.run([optimizer, total_loss, global_step],
    #                                                              feed_dict={state_ph: batch_state_seq,
    #                                                                         action_ph: batch_action,
    #                                                                         target_ph: target_rewards})
    #                     global_step_val = int(global_step_val)
    #                     if global_step_val % 100 == 0:
    #                         logger.info("training step=%d, loss=%.6f" % (global_step_val, loss))
    #                     # save network model
    #                     if global_step_val % 10000 == 0:
    #                         save_model(self.sess, self.train_dir, self.saver, "dqn", global_step=global_step_val)
    #                         logger.info("save network model, global_step=%d" % global_step_val)
    #             # update state
    #             state_seq = state_seq_n
    #             train_step += 1
    #             # state description
    #             if train_step < self.observe:
    #                 state_desc = "observe"
    #             elif epsilon > self.final_epsilon:
    #                 state_desc = "explore"
    #             else:
    #                 state_desc = "train"
    #             if reward_n > stage_reward:
    #                 stage_reward = reward_n
    #             if terminal_n:
    #                 break
    #         # record reward
    #         if stage_reward > max_reward:
    #             max_reward = stage_reward
    #         logger.info(
    #             "epoch=%d, state=%s, global_step=%d, max_reward=%d, epsilon=%.5f, reward=%d" %
    #             (epoch, state_desc, global_step, max_reward, epsilon, stage_reward))


if __name__ == "__main__":
    model = DQN(action_num=3)
    model.train()
