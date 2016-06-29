#!/usr/bin/python
#  -*- coding: utf-8 -*-
# author:  <yao62995@gmail.com> 

import threading
import numpy as np
from common import *


class A3CModel(object):
    """
        Advantage async actor-critic model.
        ref:
            paper "asynchronous methods for deep reinforcement learning"
    """

    def __init__(self, actions=225, train_dir="./a3c_models", gpu_id=0):
        self.action_num = actions

        self.learn_rate = 1e-4
        self.train_dir = train_dir
        self.gpu_id = gpu_id
        if not os.path.isdir(self.train_dir):
            os.mkdir(self.train_dir)
        self.sess, self.saver, self.graph_ops = self.build_graph()

    def policy_model(self, _input, shape, board_size=15):
        fc = full_connect(_input, (shape[0], board_size * board_size), "fc_p", activate=None)
        softmax_linear = tf.nn.softmax(fc)
        return softmax_linear

    def value_model(self, _input, shape):
        fc = full_connect(_input, (shape[0], 1), "fc_v", activate=None)
        return fc

    def inference(self, _input):
        # first conv1
        conv1 = conv2d(_input, (5, 5, 3, 32), "conv_1", stride=1)
        # norm1
        norm1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm_1')
        # conv2 ~ conv_k
        pre_layer = norm1
        for i in xrange(5):
            conv_k = conv2d(pre_layer, (3, 3, 32, 32), "conv_%d" % (i + 2), stride=1)
            norm2 = tf.nn.lrn(conv_k, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm_%d' % (i + 2))
            pre_layer = norm2
        # last layer
        conv_n = conv2d(pre_layer, (1, 1, 32, 32), "conv_n", stride=1)
        norm_n = tf.nn.lrn(conv_n, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm_n')

        reshape = tf.reshape(norm_n, [-1, 225 * 32])
        # dim = reshape.get_shape()[1].value
        logits = full_connect(reshape, (225 * 32, 1024), "fc_1")
        return logits

    def build_graph(self):
        with tf.Graph().as_default(), tf.device('/gpu:%d' % self.gpu_id):
            global_step = tf.get_variable('global_step', [],
                                          initializer=tf.constant_initializer(0), trainable=False)
            # init placeholder
            state_ph = tf.placeholder(tf.float32, [None, 15, 15, 3])
            action_ph = tf.placeholder(tf.float32, shape=[None, self.action_num])
            target_ph = tf.placeholder(tf.float32, shape=[None])
            # create model
            shared_dim = 1024
            shared = self.inference(state_ph, 3, out_dim=shared_dim)
            policy_out = self.policy_model(shared, [shared_dim])
            value_out = self.value_model(shared, [shared_dim])
            # calculate loss
            policy_loss = tf.reduce_mean(
                -tf.log(tf.reduce_mean(tf.mul(policy_out, action_ph), reduction_indices=1)) * tf.square(
                    target_ph - value_out))
            value_loss = tf.reduce_mean(tf.square(target_ph - value_out))
            l2_loss = tf.add_n(tf.get_collection('losses'), name='l2_loss')
            total_loss = policy_loss + value_loss + l2_loss
            # optimizer
            optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(total_loss, global_step=global_step)
            # optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(total_loss, global_step=global_step)
            saver = tf.train.Saver()
            init = tf.initialize_all_variables()
            sess = tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False)
            )
            sess.run(init)
            # restore model
            restore_model(sess, self.train_dir, saver)
            graph_ops = state_ph, action_ph, target_ph, optimizer, policy_out, value_out, total_loss, global_step
            return sess, saver, graph_ops

    def train(self, thread_num=3):
        envs = [Environment() for _ in xrange(thread_num)]
        actor_learner_threads = []
        t_max, gamma = 64, 0.99
        constants = self.action_num, t_max, gamma, self.train_dir
        for idx in xrange(envs):
            _thread = threading.Thread(target=actor_learner_thread,
                                       args=(idx, envs[idx], self.sess, self.saver, self.graph_ops, constants))
            _thread.start()
            actor_learner_threads.append(_thread)
        for _thread in actor_learner_threads:
            _thread.join()


def sample_action(probs):
    probs = probs - np.finfo(np.float32).epsneg
    histogram = np.random.multinomial(1, probs)
    action_index = int(np.nonzero(histogram)[0])
    return action_index


def actor_learner_thread(thread_id, env, sess, saver, graph_ops, constants):
    state_ph, action_ph, target_ph, optimizer, policy_out, value_out, total_loss, global_step = graph_ops
    action_num, t_max, gamma, train_dir = constants
    terminal = False
    episode_reward = 0
    episode_step = 0
    episode_loss = []
    episode_count = 0
    state = env.get_state()
    start_time = time.time()
    while True:
        states, actions, rewards = [], [], []
        t_start = episode_step
        while not terminal and ((episode_step - t_start) < t_max):
            probs = sess.run([policy_out], feed_dict={state_ph: [state]})[0][0]
            action = sample_action(probs)
            state_n, reward_n, terminal = env.step_forward(action)
            one_hot_action = np.zeros(action_num)
            one_hot_action[action] = 1
            states.append(state)
            actions.append(one_hot_action)
            rewards.append(reward_n)
            state = state_n
            episode_step += 1
            episode_reward += reward_n
        rewards_R = np.zeros(len(rewards))
        if not terminal:
            R = sess.run([value_out], feed_dict={state_ph: [state]})[0][0]
        else:
            R = 0
        for t_idx in xrange(episode_step - t_start - 1, -1, -1):
            R = rewards[t_idx] + gamma * R
            rewards_R[t_idx] = R
        _, loss, gt = sess.run([optimizer, total_loss, global_step], feed_dict={state_ph: states, action_ph: actions,
                                                                                target_ph: rewards_R})
        episode_loss.append(float(loss))
        gt = int(gt)
        if gt % 500 == 0:
            save_model(sess, train_dir, saver, "policy_rl_a3c", global_step=gt)
        if terminal:
            elapsed_time = int(time.time() - start_time)
            env.reset()
            state = env.get_state()
            terminal = False
            episode_loss = sum(episode_loss) / len(episode_loss)
            episode_count += 1
            logger.info("thread=%d, T=%d, episode(count=%d, step=%d, loss=%.5f, reward=%d), time=%d(s)" %
                        (thread_id, gt, episode_count, episode_step, episode_loss, episode_reward, elapsed_time))
            start_time = time.time()
            episode_reward = 0
            episode_step = 0
            episode_loss = []


if __name__ == "__main__":
    model = A3CModel()
    model.train(thread_num=3)
