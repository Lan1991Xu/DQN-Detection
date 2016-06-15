import tensorflow as tf
import numpy as np
import time
import random

from .ops import cov_layer, fc_layer
from .memory import Memory
from .environtment import Environment, State
from .base import BaseModel
from config import Config

class Agent(BaseModel):
    def __init__(self, config, sess):
        """Agent initialization
            
            initialize the agent, including mem, env, action_history, action_status, session

        """
        super(Agent, self).__init__(config)
        self.action_history = tf.placeholder('float32', [None, config.action_size], 'action_history')
        self.build_cnn_net(False)
        self.build_cnn_net(True)
        self.build_dqn_net(False)
        self.build_dqn_net(True)
        self.mem = Memory(config.mem_capacity)
        self.env = Environment(config, sess)
        self.action_status = 0
        self.sess = sess

    def build_cnn_net(self, target):
        """build cnn_net part

        build the 5 conv layers cnn to extract features.

        """
        # initializer, rectifier and normalizer
        activation = tf.nn.relu
        w_initializer = tf.truncated_normal_initializer(self.ini_mean, self.ini_stddev)
        b_initializer = tf.constant_initializer(self.bias_starter)

        if target:
            scope_name = 't_CNN'
            self.t_cnn_w = {}
            cur_w = self.t_cnn_w 
            self.t_inp = tf.placeholder('float32', [None, 224, 224, 3], name = 't_inp')
            inp = self.t_inp
        else:
            scope_name = 'p_CNN'
            self.p_cnn_w = {}
            cur_w = self.p_cnn_w
            self.p_inp = tf.placeholder('float32', [None, 224, 224, 3], name = 'p_inp')
            inp = self.p_inp

        with tf.variable_scope(scope_name):
            # CNN_l1(including pooling and normlization)
            l1, cur_w['l1_w'], cur_w['l1_b'] = cov_layer(inp, 96, [7, 7], [2, 2], w_initializer = w_initializer, b_initializer = b_initializer, activation = activation, padding = 'VALID', name = 'cnn_conv1')
            pool1 = tf.nn.max_pool(l1, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'VALID', name = 'pool1')

            # CNN_l2(including pooling and normlization)
            l2, cur_w['l2_w'], cur_w['l2_b'] = cov_layer(pool1, 256, [5, 5], [2, 2], w_initializer = w_initializer, b_initializer = b_initializer, activation = activation, padding = 'VALID', name = 'cnn_conv2')
            pool2 = tf.nn.max_pool(l2, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'VALID', name = 'pool2')
            
            # CNN_l3
            l3, cur_w['l3_w'], cur_w['l3_b'] = cov_layer(pool2, 384, [3, 3], [1, 1], w_initializer = w_initializer, b_initializer = b_initializer, activation = activation, padding = 'VALID', name = 'cnn_conv3')

            # CNN_l4
            l4, cur_w['l4_w'], cur_w['l4_b'] = cov_layer(l3, 384, [3, 3], [1, 1], w_initializer = w_initializer, b_initializer = b_initializer, activation = activation, padding = 'VALID', name = 'cnn_conv4')

            # CNN_l5
            l5, cur_w['l5_w'], cur_w['l5_b'] = cov_layer(l4, 256, [3, 3], [1, 1], w_initializer = w_initializer, b_initializer = b_initializer, activation = activation, padding = 'VALID', name = 'cnn_conv5')
            pool5 = tf.nn.max_pool(l5, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'VALID', name = 'pool5')

            # CNN_l5 reshape
            shape = pool5.get_shape().as_list()
            l5_flat = tf.reshape(pool5, [-1, reduce(lambda x, y: x * y, shape[1:])], name = 'l5_flat')
            
            # CNN_output
            out, cur_w['output_w'], cur_w['output_b'] = fc_layer(l5_flat, 4096, activation = activation, w_initializer = w_initializer, b_initializer = b_initializer, name = 'output')

            if target:
                self.t_cnn_out = out
            else:
                self.p_cnn_out = out

        if target:
            with tf.variable_scope('cnn_transfer'):
                self.cnn_assign_inp = {}
                self.cnn_assign_op = {}

                for key in self.p_cnn_w.keys():
                    self.cnn_assign_inp[key] = tf.placeholder('float32', self.p_cnn_w[key].get_shape().as_list(), name = key)
                    self.cnn_assign_op[key] = self.t_cnn_w[key].assign(self.cnn_assign_inp[key])


    def build_dqn_net(self, target):
        """build dqn part

        build the 2 fc layers dqn to q_function.

        """
        # initializer and rectifier
        activation = tf.nn.relu
        w_initializer = tf.truncated_normal_initializer(self.ini_mean, self.ini_stddev)
        b_initializer = tf.constant_initializer(self.bias_starter)

        # inp_size = config.featureDimension + config.actionDimension
        
        if target:
            name_scope = 't_DQN'
            self.t_dqn_w = {}
            cur_w = self.t_dqn_w
            inp = self.t_cnn_out
        else:
            name_scope = 'p_DQN'
            self.p_dqn_w = {}
            cur_w = self.p_dqn_w
            inp = self.p_cnn_out

        inp = tf.concat(1, [inp, self.action_history], name = name_scope + '_concat')  

        with tf.variable_scope(name_scope):
            # DQN_fc1
            l1, cur_w['l1_w'], cur_w['l1_b'] = fc_layer(inp, 1024, activation = activation, w_initializer = w_initializer, b_initializer = b_initializer, name = 'dqn_l1')
            
            # DQN_fc2
            l2, cur_w['l2_w'], cur_w['l2_b'] = fc_layer(l1, 1024, activation = activation, w_initializer = w_initializer, b_initializer = b_initializer, name = 'dqn_l2')

            # DQN_output
            out, cur_w['output_w'], cur_w['output_b'] = fc_layer(l2, self.action_size, w_initializer = w_initializer, b_initializer = b_initializer, name = 'dqn_q')
        
            if target:
                self.t_q = out
            else:
                self.p_q = out

            if not target:
                self.q_action = tf.argmax(out, dimension = 1, name = 'q_action')

        if not target:
            # optimizer
            with tf.variable_scope('dqn_optimizer'):
                self.dqn_gt_q = tf.placeholder('float32', [None], name = 'dqn_gt_q')
                self.action = tf.placeholder('int64', [None], name = 'action')

                action_one_hot = tf.one_hot(self.action, self.action_size, 1.0, 0.0, name = 'action_one_hot')
                q_acted = tf.reduce_sum(self.p_q * action_one_hot, reduction_indices = 1, name = 'q_acted')

                self.dqn_delta = self.dqn_gt_q - q_acted
                self.clipped_delta = tf.clip_by_value(self.dqn_delta, self.min_delta, self.max_delta, name = 'clipped_delta')
                # self.global_step = tf.Varialbe(0, trainable = False)

                self.dqn_loss = tf.reduce_mean(tf.square(self.clipped_delta), name = 'dqn_loss')
                self.dqn_learning_rate_step = tf.placeholder('int64', None, name = 'learning_rate_step')
                self.dqn_learning_rate_op = tf.maximum(self.learning_rate_minimum,
                        tf.train.exponential_decay(
                            self.dqn_learning_rate,
                            self.dqn_learning_rate_step,
                            self.dqn_learning_rate_decay_step,
                            self.dqn_learning_rate_decay,
                            staircase = True))
                self.dqn_optim = tf.train.RMSPropOptimizer(self.dqn_learning_rate_op, momentum = self.dqn_momentum, epsilon = self.dqn_epsilon).minimize(self.dqn_loss)
        else:
            with tf.variable_scope('dqn_transfer'):
                self.dqn_assign_inp = {}
                self.dqn_assign_op = {}

                for key in self.p_dqn_w.keys():
                    self.dqn_assign_inp[key] = tf.placeholder('float32', self.p_dqn_w[key].get_shape().as_list(), name = key)
                    self.dqn_assign_op[key] = self.t_dqn_w[key].assign(self.dqn_assign_inp[key])

    def train(self):
        # DQN initialization
        
        # timer
        st = time.time()

        self.sess.run(tf.initialize_all_variables())
        # timer
        print "Spent %.4fsecs initializing..." % (time.time() - st)
        st = time.time()
        # To be uncommented
        # self.update_target_net()
        # timer
        print "Spent %.4fsecs assigning..." % (time.time() - st)
        st = time.time()
        #
        self.ep_rewards = []
        self.update_count = 0
        self.mem.reset()
        # timer
        print "Spent %.4fsecs resetting..." % (time.time() - st)
        st = time.time()
        #
        self.step = 0

        # timer
        print "Spent %.4fsecs initializing..." % (time.time() - st)
        #
        data_size = self.env.get_size('train')
        if data_size > self.epi_size:
            self.epi_size = data_size

        # start the env.dataset.readerqueue
        self.env.start_train()

        for episode in xrange(self.epi_size):
            # initialize the environment for each episode
            state = self.env.reset()   
            state = State(state.img, state.height, state.width)
            # for x in xrange(self.mem_capacity):
                # self.mem.add(state)
            self.action_status = 0
            
            for stp in xrange(self.step_size):
                self.step += 1
                # predict
                action = self.predict(np.array([self.env.state]))
                # Debug
                print "Done prediction: Ep %d, Step %d" % (episode, stp)
                #
                # act
                nxt_state, reward, terminal = self.env.act(action)
                # Debug
                print "Done action: Ep %d, Step %d" % (episode, stp)
                #
                # observe
                self.observe(state, action, reward, nxt_state, terminal)
                # Debug
                print "Done observe: Ep %d, Step %d" % (episode, stp)
                #
                # exit()

                if terminal:
                    break
                else:
                    state = nxt_state
                    self.action_status |= 1 << action

                # info
                print "Trained on episode %d, step %d" % (episode, stp)

            if episode % self.check_point == 0:
                self.evaluation()

        # close the env.dataset.readerqueue
        self.env.end_train()

    def predict(self, states):
        if random.random() < self.act_ep:
            action = random.randrange(self.env.action_size)
        else:
            action = self.sess.run(self.q_action, {self.action_history : self.actionArray(1), self.p_inp: self.crop(states)})
            action = action[0]
        return action

    def observe(self, state, action, reward, nxt_state, terminal):
        # clip reward
        reward = max(self.min_reward, min(self.max_reward, reward)) 

        self.mem.add(state, action, reward, nxt_state, terminal)
        if self.step < self.learning_start_point:
            return

        # gradient descent
        self.mini_batch_gradient_descent()
        # update target net every C steps
        if self.step % self.update_C == 0:
            self.update_target_net()

    def mini_batch_gradient_descent(self):
        if self.mem.count < self.batch_size:
            return

        s, action, reward, s_nxt, terminal = self.mem.sample(self.batch_size)

        q_nxt = self.sess.run(self.t_q, {self.t_inp : self.crop(s_nxt), self.action_history : self.actionArray(self.batch_size)})
        
        terminal = np.array(terminal) + 0.
        max_q_nxt = np.max(q_nxt, axis = 1)
        ground_truth = (1. - terminal) * self.discount * max_q_nxt + reward

        _, q_t, loss = self.sess.run([self.dqn_optim, self.p_q, self.dqn_loss], {
                self.dqn_gt_q : ground_truth,
                self.action : action,
                self.p_inp : self.crop(s),
                self.dqn_learning_rate_step: self.step,
                self.action_history: self.actionArray(self.batch_size)
                })

        self.update_count += 1
        # DEBUG
        print "Update_count : %d" % self.update_count, loss

    def actionArray(self, sz):
        arr = np.zeros([sz, self.action_size], dtype = float)
        act1d = np.empty([self.action_size], dtype = float)
        for x in xrange(self.action_size):
            if (self.action_status >> x) & 1 == 1:
                act1d[x] = 1.0
            else:
                act1d[x] = 0.0 
        return arr + act1d

    def update_target_net(self):
        for key in self.p_cnn_w.keys():
            self.sess.run(self.cnn_assign_op[key], {self.cnn_assign_inp[key] : self.sess.run(self.p_cnn_w[key])})
        for key in self.p_dqn_w.keys():
            self.sess.run(self.dqn_assign_op[key], {self.dqn_assign_inp[key] : self.sess.run(self.p_dqn_w[key])})

    def crop(self, states):
        # Debug
        # print states
        #
        cropped = np.empty([states.shape[0], 224, 224, 3], dtype = np.float32)
        cnt = 0

        for s in states:
            img = s.img
            up = s.box[0]
            left = s.box[1]
            down = s.box[2]
            right = s.box[3]
            patch = tf.constant(img[up - 1 : down, left - 1 : right, :])
            resized_patch = tf.image.resize_image_with_crop_or_pad(patch, 224, 224)
            casted_patch = tf.cast(resized_patch, dtype = tf.float32)
            cropped[cnt] = self.sess.run(casted_patch)
            cnt += 1

        return cropped

    def evaluation(self):
        pass 

    def record(self):
        pass
