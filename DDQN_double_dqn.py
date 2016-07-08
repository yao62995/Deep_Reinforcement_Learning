#!/usr/bin/python
#  -*- coding: utf-8 -*-
# author: yao62995@gmail.com

from collections import deque

from common import *


class DDQN(Base):
    """
        Double Deep Q-learning
        ref:
            paper "Deep Reinforcement Learning with Double Q-learning"
    """

    def __init__(self, states_dim, actions_dim, action_range=(-1, 1), train_dir="./ddqn_models", gpu_id=0,
                 observe=1e3, replay_memory=5e4, update_frequency=1, train_repeat=1, frame_seq_num=1, gamma=0.99,
                 batch_size=64, learn_rate=1e-3, update_target_freq=64):
        Base.__init__(self)
        self.states_dim = states_dim
        self.actions_dim = actions_dim
        self.action_range = action_range
        self.gpu_id = gpu_id
        self.frame_seq_num = frame_seq_num
        self.update_target_freq = update_target_freq
        # init train params
        self.observe = observe
        self.update_frequency = update_frequency
        self.train_repeat = train_repeat
        self.gamma = gamma
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
        self.weight = {"q": [], 't': []}
        self.bias = {"q": [], 't': []}
        # build graph
        self.build_graph()
        # assign q-network to target network
        self.update_target_network()

    def append_params(self, scope_name, w, b):
        self.weight[scope_name].append(w)
        self.bias[scope_name].append(b)

    def update_target_network(self):
        for idx in xrange(len(self.weight["q"])):
            self.sess.run(self.weight["t"][idx].assign(self.weight["q"][idx]))
        for idx in xrange(len(self.bias["q"])):
            self.sess.run(self.bias["t"][idx].assign(self.bias["q"][idx]))

    def inference(self, variable_scope, state, weight_decay=None):
        with tf.variable_scope(variable_scope) as scope:
            self.ops[scope.name]["state"] = state
            # conv layer 1 with max pool
            conv1, w, b = conv2d(state, (8, 8, 3 * self.frame_seq_num, 32), "conv1", stride=2,
                                 with_param=True, weight_decay=weight_decay)
            pool1 = max_pool(conv1, ksize=2, stride=2, name="pool1")
            self.append_params(scope.name, w, b)
            # conv layer 2 with avg pool
            conv2, w, b = conv2d(pool1, (3, 3, 32, 32), "conv2", stride=2, with_param=True, weight_decay=weight_decay)
            self.append_params(scope.name, w, b)
            pool2 = avg_pool(conv2, ksize=2, stride=2, name="pool2")
            # reshape
            flat1 = tf.reshape(pool2, (-1, 16 * 32), name="flat1")
            # fc1
            fc1, w, b = full_connect(flat1, (16 * 32, 256), "fc1", with_param=True, weight_decay=weight_decay)
            self.append_params(scope.name, w, b)
            # out
            logits, w, b = full_connect(fc1, (256, self.action_num), "out", activate=None,
                                        with_param=True, weight_decay=weight_decay)
            self.append_params(scope.name, w, b)
            return logits

    def build_graph(self):
        with tf.Graph().as_default(), tf.device('/gpu:%d' % self.gpu_id):
            # set global step
            self.global_step = tf.get_variable('global_step', [],
                                               initializer=tf.constant_initializer(0), trainable=False)
            # init q-network
            state = tf.placeholder(tf.float32, shape=(None, 64, 64, 3 * self.frame_seq_num), name="state")
            logits = self.inference("q", state, weight_decay=1e-2)
            # loss
            action = tf.placeholder(tf.float32, shape=(None, self.actions_dim), name="action")
            q_target = tf.placeholder(tf.float32, shape=(None), name="q_target")
            l2_loss = tf.add_n(tf.get_collection("losses"))
            q_loss = tf.reduce_mean(tf.square(tf.reduce_mean(tf.mul(logits, action), reduction_indices=1) - q_target))
            total_loss = q_loss + l2_loss
            # optimizer
            train_opt = tf.train.AdamOptimizer(self.learn_rate).minimize(total_loss, global_step=self.global_step)

            # init target q-network
            state2 = tf.placeholder(tf.float32, shape=(None, 64, 64, 3 * self.frame_seq_num), name="target_state")
            logits2 = self.inference("t", state2)
            # init session and saver
            self.saver = tf.train.Saver(max_to_keep=5)
            self.sess = tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False)
            )
            self.sess.run(tf.initialize_all_variables())
        # restore model
        restore_model(self.sess, self.train_dir, self.saver)
        # some ops
        self.ops["logits"] = lambda obs: self.sess.run([logits], feed_dict={state: obs})
        self.ops["logits_target"] = lambda obs: self.sess.run([logits2], feed_dict={state2: obs})
        self.ops["train_q"] = lambda obs, act, q_t: self.sess.run([train_opt, total_loss, self.global_step],
                                                                  feed_dict={state: obs, action: act, q_target: q_t})

    def get_action(self, state, with_noise=False):
        action = self.ops["logits"]([state])[0][0]
        if with_noise:
            action = np.clip(action + self.explore_noise.noise(), self.action_range[0], self.action_range[1])
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
