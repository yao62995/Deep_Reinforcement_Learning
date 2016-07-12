#!/usr/bin/python
#  -*- coding: utf-8 -*-
# author: yao62995@gmail.com

import numpy as np
from collections import deque

from common import *

# from tensorflow.python.framework import ops
# @ops.RegisterGradient("norm_adv")
# def normalized_advantange(op, grad, some_other_arg):

class NAF(Base):
    """
        Normalized Advantage Function
        ref:
            paper "Continuous Deep Q-Learning with Model-based Acceleration"
    """

    def __init__(self, states_dim, actions_dim, train_dir="./ddpg_models", gpu_id=0,
                 observe=1e3, replay_memory=5e4, update_frequency=1, train_repeat=1, gamma=0.99, tau=0.001,
                 batch_size=64, learn_rate=1e-3, dim=256):
        Base.__init__(self)
        self.states_dim = states_dim
        self.actions_dim = actions_dim
        self.gpu_id = gpu_id
        # init train params
        self.observe = observe
        self.update_frequency = update_frequency
        self.train_repeat = train_repeat
        self.gamma = gamma
        self.tau = tau
        # init replay memory deque
        self.replay_memory_size = replay_memory
        self.replay_memory = deque()
        # init noise
        self.explore_noise = OUNoise(self.actions_dim)
        # train models dir
        self.train_dir = train_dir
        if not os.path.isdir(self.train_dir):
            os.mkdir(self.train_dir)
        # init network params
        self.learn_rate = learn_rate
        self.batch_size = batch_size
        # tensorflow graph variables
        self.sess = None
        self.saver = None
        self.global_step = None
        self.ops = dict()
        # build graph
        self.build_graph(dim=dim)

    def target_exponential_moving_average(self, theta):
        ema = tf.train.ExponentialMovingAverage(decay=1 - self.tau)
        update = ema.apply(var_list=theta)
        averages = [ema.average(x) for x in theta]
        return averages, update

    def get_variables(self, scope, shape, wd=0.01, val_range=None, collect="losses"):
        with tf.variable_scope(scope):
            if val_range is None:
                val_range = (-1 / np.sqrt(shape[0]), 1 / np.sqrt(shape[0]))
            weights = tf.Variable(tf.random_uniform(shape, val_range[0], val_range[1]), name='weights')
            biases = tf.Variable(tf.random_uniform([shape[-1]], val_range[0], val_range[1]), name='biases')
            if wd is not None:
                weight_decay = tf.mul(tf.nn.l2_loss(weights), wd, name='weight_loss')
                tf.add_to_collection(collect, weight_decay)
            return weights, biases

    def network_variables(self, scope, wd=0.01, dim=256):
        with tf.variable_scope(scope):
            h1_w, h1_b = self.get_variables("fc1", (self.states_dim, dim), wd=wd, collect=scope)
            h2_w, h2_b = self.get_variables("fc2", (dim, dim), wd=wd, collect=scope)
            h3_w, h3_b = self.get_variables("fc3", (dim, dim), wd=wd, collect=scope)
            # value variable
            v_w, v_b = self.get_variables("fc_v", (dim, 1), val_range=(-3e-4, 3e-4), wd=wd, collect=scope)
            # μ variable
            mu_w, mu_b = self.get_variables("fc_mu", (dim, self.actions_dim), val_range=(-3e-4, 3e-4), wd=wd, collect=scope)
            # lower-triangular matrix
            l_w, l_b = self.get_variables("fc_l", (dim, self.actions_dim * (self.actions_dim + 1) / 2),
                                          val_range=(-3e-4, 3e-4), wd=wd, collect=scope)
            return [h1_w, h1_b, h2_w, h2_b, h3_w, h3_b,
                    v_w, v_b, mu_w, mu_b, l_w, l_b]

    def lower_triangular(self, x, n):
        """
        :param x: a tensor with shape (batch_size, n*(n+1)/2)
        :return: a tensor of lower-triangular with shape (batch_size, n, n)
        """
        x = tf.transpose(x, perm=(1, 0))
        # target = tf.Variable(np.zeros((n * n, self.batch_size)), dtype=tf.float32)
        target = tf.Variable(np.zeros((n * n, self.batch_size)), dtype=tf.float32)
        # update diagonal values
        diag_indics = tf.square(tf.range(n))
        t1 = tf.stop_gradient(tf.scatter_update(target, diag_indics, x[:n, :]))
        # update lower values
        u, v = np.tril_indices(n, -1)
        lower_indics = tf.constant(u * n + v)
        t2 = tf.stop_gradient(tf.scatter_update(target, lower_indics, x[n:, :]))
        # reshape lower matrix to lower-triangular matrix
        target = tf.add(t1, t2)
        target = tf.transpose(target, (1, 0))
        target = tf.reshape(target, (self.batch_size, n, n))
        return target

    def inference(self, op_scope, state, action, theta, batch_norm=True):
        h1_w, h1_b, h2_w, h2_b, h3_w, h3_b = theta[:6]
        v_w, v_b, mu_w, mu_b, l_w, l_b = theta[6:]
        with tf.variable_op_scope([state, action], op_scope, "naf"):
            # full connect layers
            fc1 = tf.nn.relu(tf.matmul(state, h1_w) + h1_b)
            if batch_norm:
                fc1 = NetTools.batch_normalized(fc1)
            fc2 = tf.nn.relu(tf.matmul(fc1, h2_w) + h2_b)
            if batch_norm:
                fc2 = NetTools.batch_normalized(fc2)
            fc3 = tf.nn.relu(tf.matmul(fc2, h3_w) + h3_b)
            # value layer
            h_v = tf.add(tf.matmul(fc3, v_w), v_b, name="value")
            # μ layer
            h_mu = tf.add(tf.matmul(fc3, mu_w), mu_b, name="miu")
            # Advantage layer: step_1 - full connect to linear layer
            h_l = tf.matmul(fc3, l_w) + l_b
            # transform (batch_size, n*(n+1)) to (batch_size, n, n) with low_triangular format
            l1_var = self.lower_triangular(h_l, self.actions_dim)
            # Advantage layer: step_3 - positive-definite square matrix
            p_var = tf.batch_matmul(l1_var, tf.transpose(l1_var, perm=[0, 2, 1]))
            diff_u = tf.reshape(action - h_mu, (-1, 1, self.actions_dim))
            # a_var = (diff_u * p_val * diff_u.T)
            a_var = tf.batch_matmul(tf.batch_matmul(diff_u, p_var), tf.transpose(diff_u, perm=[0, 2, 1]))
            a_var = tf.squeeze(a_var, [1], name='advantage')
            q = tf.add(h_v, a_var)
            return h_mu, h_v, q

    def build_graph(self, dim=256):
        with tf.Graph().as_default(), tf.device('/gpu:%d' % self.gpu_id):
            self.global_step = tf.get_variable('global_step', [],
                                               initializer=tf.constant_initializer(0), trainable=False)
            # init variables
            theta_q = self.network_variables("naf", dim=dim)
            theta_qt, update_qt = self.target_exponential_moving_average(theta_q)
            # network
            state = tf.placeholder(tf.float32, shape=(None, self.states_dim), name="state")
            action = tf.placeholder(tf.float32, shape=(None, self.actions_dim), name="action")
            q_target = tf.placeholder(tf.float32, shape=(None), name="q_target")
            mu, v, q = self.inference("network", state, action, theta_q)
            # target network
            state_t = tf.placeholder(tf.float32, shape=(None, self.states_dim), name="state_t")
            action_t = tf.placeholder(tf.float32, shape=(None, self.actions_dim), name="action_t")
            mu_t, v_t, q_t = self.inference("target_network", state_t, action_t, theta_qt)
            # loss
            l2_loss = tf.add_n(tf.get_collection("naf"))
            q_loss = tf.reduce_mean(tf.square(q - q_target)) + l2_loss
            # optimizer
            opt = tf.train.AdamOptimizer(self.learn_rate).minimize(q_loss, global_step=self.global_step)
            with tf.control_dependencies([opt]):
                train_q = tf.group(update_qt)
            # init session and saver
            self.saver = tf.train.Saver()
            self.sess = tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False)
            )
            self.sess.run(tf.initialize_all_variables())
        # restore model
        restore_model(self.sess, self.train_dir, self.saver)
        self.ops["act_logit"] = lambda obs: self.sess.run(mu, feed_dict={state: obs})
        self.ops["target_value"] = lambda obs: self.sess.run(v_t, feed_dict={state_t: obs})
        self.ops["train_q"] = lambda obs, act, q_val: self.sess.run([train_q, self.global_step, q_loss],
                                                                    feed_dict={state: obs, action: act,
                                                                               q_target: q_val})

    def get_action(self, state, with_noise=False):
        action = self.ops["act_logit"]([state])[0]
        if with_noise:
            action = action + self.explore_noise.noise()
        return action

    def feedback(self, state, action, reward, terminal, state_n):
        self.time_step += 1
        self.replay_memory.append((state, action, reward, terminal, state_n))
        if len(self.replay_memory) > self.replay_memory_size:
            self.replay_memory.popleft()
        if self.time_step > self.observe and self.time_step % self.update_frequency == 0:
            for _ in xrange(self.train_repeat):
                # train mini-batch from replay memory
                mini_batch = random.sample(self.replay_memory, self.batch_size)
                batch_state = [sample[0] for sample in mini_batch]
                batch_action = [sample[1] for sample in mini_batch]
                batch_state_n = [sample[4] for sample in mini_batch]
                batch_target_value = self.ops["target_value"](batch_state_n)
                for batch_i, sample in enumerate(mini_batch):
                    b_reward, b_terminal = sample[2], sample[3]
                    if b_terminal:
                        batch_target_value[batch_i] = b_reward
                    else:  # compute target q values
                        batch_target_value[batch_i] = b_reward + self.gamma * [0][0]
                # update critic network (theta_q)
                _, global_step, q_loss = self.ops["train_q"](batch_state, batch_action, batch_target_value)
                if self.time_step % 1e3 == 0:
                    # logger.info("step=%d, p_loss=%.6f, q_loss=%.6f" % (global_step, p_loss, q_loss))
                    logger.info("step=%d, q_loss=%.6f" % (global_step, q_loss))
        if self.time_step % 3e4 == 0:
            save_model(self.sess, self.train_dir, self.saver, "naf-", global_step=self.global_step)
