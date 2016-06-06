import tensorflow as tf
import numpy as np

from .ops import cov_layer, linear_layer

class Agent(BaseModel):
    def __init__(self,config):
        super(Agent, self).__init__(config)
        self.build_net(config)

    def build_cnn_net(self,config):
        """Build the pre-CNN

        Args:
            config: the global configer which offers all of required configuration.
        """
        
        # initializer, rectifier and normalizer
        activation = tf.nn.relu
        w_initializer = tf.truncated_normal_initializer(config.ini_mean, config.ini_stddev)
        b_initializer = tf.constant_initializer(config.bias_starter)

        self.cnn_w = {}

        with tf.variable_scope('CNN'):
            # Input_Holder
            inp = tf.placeholder(config.imgDType, [None, 224, 224, 3], 'ZFNet_input')
            # CNN_l1(including pooling and normlization)
            self.cnn_l1, self.cnn_w['l1_w'], self.cnn_w['l1_b'] = cov_layer(inp, 96, [7, 7], [2, 2], w_initializer = w_initializer, b_initializer = b_initializer, activation = activation, name = 'cnn_conv1')
            self.pool1 = tf.nn.max_pool(self.cnn_l1, ksize = [1, 3, 3, 1], stride = [1, 2, 2, 1], padding = 'SAME', name = 'pool1')

            # CNN_l2(including pooling and normlization)
            self.cnn_l2, self.cnn_w['l2_w'], self.cnn_w['l2_b'] = cov_layer(self.pool1, 256, [5, 5], [2, 2], w_initializer = w_initializer, b_initializer = b_initializer, activation = activation, name = 'cnn_conv2')
            self.pool2 = tf.nn.max_pool(self.cnn_l2, ksize = [1, 3, 3, 1], stride = [1, 2, 2, 1], padding = 'SAME', name = 'pool2')
            
            # CNN_l3
            self.cnn_l3, self.cnn_w['l3_w'], self.cnn_w['l3_b'] = cov_layer(self.pool2, 384, [3, 3], [1, 1], w_initializer = w_initializer, b_initializer = b_initializer, activation = activation, name = 'cnn_conv3')

            # CNN_l4
            self.cnn_l4, self.cnn_w['l4_w'], self.cnn_w['l4_b'] = cov_layer(self.cnn_l3, 384, [3, 3], [1, 1], w_initializer = w_initializer, b_initializer = b_initializer, activation = activation, name = 'cnn_conv4')

            # CNN_l5
            self.cnn_l5, self.cnn_w['l5_w'], self.cnn_w['l5_b'] = cov_layer(self.cnn_l4, 256, [3, 3], [1, 1], w_initializer = w_initializer, b_initializer = b_initializer, activation = activation, name = 'cnn_conv5')
            self.pool5 = tf.nn.max_pool(self.cnn_l5, ksize = [1, 3, 3, 1], stride = [1, 2, 2, 1], padding = 'SAME', name = 'pool5')

            # CNN_l5 reshape
            self.cnn_l5_flat = tf.reshape(self.pool5, [config.batch_size, -1], name = 'l5_flat')
            
            # CNN_output
            self.cnn_output, self.cnn_w['output_w'], self.cnn_w['output_b'] = fc_layer(self.cnn_l5_flat, 4096, activation = activation, w_initializer = w_initializer, b_initializer = b_initializer, name = 'output')


    def build_dqn_net(self,config):
        """Build the Deep-Q-Net

        Args:
            config: the global configer which offers all of required configuration.

        """
        # initializer and rectifier
        activation = tf.nn.relu
        w_initializer = tf.truncated_normal_initializer(config.ini_mean, config.ini_stddev)
        b_initializer = tf.constant_initializer(config.bias_start)

        # deep-Q-net training weights(w)
        self.dqn_w = {}
        # Input_Holder
        inp_size = config.featureDimension + config.actionDimension
        inp = tf.placeholder(config.featureDType, [None, inp_size], 'dqn_input')

        with tf.variable_scope('DQN'):
            # DQN_fc1
            self.dqn_l1, self.dqn_w['l1_w'], self.dqn_b['l1_b'] = fc_layer(inp, 1024, activation = activation, w_initializer = w_initializer, b_initializer = b_initializer, name = 'dqn_l1')
            
            # DQN_fc2
            self.dqn_l2, self.dqn_w['l2_w'], self.dqn_b['l2_b'] = fc_layer(self.dqn_l1, 1024, w_initializer = w_initializer, b_initializer = b_initializer, name = 'dqn_l2')

            # DQN_output
            self.dqn_l3, self.dqn_w['output_w'], self.dqn_b['output_b'] = fc_layer(self.dqn_l2, config.action_size, w_initializer = w_initializer, b_initializer = b_initializer, name = 'dqn_l3')
            self.q_action = tf.argmax(self.q, dimension = 1)

        # optimizer
        with tf.variable_scope('dqn_optimizer'):
            self.dqn_gt_q = tf.placeholder('float32', [None], name = 'dqn_gt_q')
            self.action = tf.placeholder('int64', [None], name = 'action')

            action_one_hot = tf.one_hot(self.action, config.action_size, 1.0, 0.0, name = 'action_one_hot')
            q_acted = tf.reduce_sum(self.q * action_one_hot, reduction_indices = 1, name = 'q_acted')

            self.dqn_delta = self.dqn_gt_q - q_acted
            self.clipped_delta = tf.clip_by_value(self.delta, self.min_delta, self.max_delta, name = 'clipped_delta')
            self.global_step = tf.Varialbe(0, trainable = False)

            self.dqn_loss = tf.reduce_mean(tf.square(self.clipped_delta), name = 'dqn_loss')
            self.dqn_learning_rate_step = tf.placeholder('int64', None, name = 'learning_rate_step')
            self.dqn_learning_rate_op = tf.maximum(self.learning_rate_minimum,
                    tf.train.exponential_decay(
                        self.dqn_learning_rate_op,
                        self.dqn_learning_rate_step,
                        self.dqn_learning_rate_decay_step,
                        self.dqn_learning_rate_decay,
                        staircase = True))
            self.dqn_optim = tf.train.RMSPropOptimizer(self.learning_rate_op, momentum = config.dqn_momentum, epsilon = config.dqn_epsilon).minimize(self.loss)
        
        # DQN initialization
        tf.initialize_all_variables().run()

    def record(self):

    def train(self):

    def evaluation(self):

