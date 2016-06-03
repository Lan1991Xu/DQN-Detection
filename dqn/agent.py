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
        initializer = tf.truncated_normal_initializer(config.ini_mean, config.ini_stddev)

        self.cnn_w = {}

        with tf.variable_scope('CNN-ZFNet'):
            # Input_Holder
            inp = tf.placeholder(config.imgDType, [None, 224, 224, 3], 'ZFNet_input')
            # CNN_l1(including pooling and normlization)
            self.cnn_l1, self.cnn_w['l1_w'], self.cnn_w['l1_b'] = cov_layer(inp, 96, [7, 7], [2, 2], initializer = initializer, activation = activation, name = 'cnn_conv1')
            self.pool1 = tf.nn.max_pool(self.cnn_l1, ksize = [1, 3, 3, 1], stride = [1, 2, 2, 1], padding = 'SAME', name = 'pool1')

            # CNN_l2(including pooling and normlization)
            self.cnn_l2, self.cnn_w['l2_w'], self.cnn_w['l2_b'] = cov_layer(self.pool1, 256, [5, 5], [2, 2], initializer = initializer, activation = activation, name = 'cnn_conv2')
            self.pool2 = tf.nn.max_pool(self.cnn_l2, ksize = [1, 3, 3, 1], stride = [1, 2, 2, 1], padding = 'SAME', name = 'pool2')
            
            # CNN_l3
            self.cnn_l3, self.cnn_w['l3_w'], self.cnn_w['l3_b'] = cov_layer(self.pool2, 384, [3, 3], [1, 1], initializer = initializer, activation = activation, name = 'cnn_conv3')

            # CNN_l4
            self.cnn_l4, self.cnn_w['l4_w'], self.cnn_w['l4_b'] = cov_layer(self.cnn_l3, 384, [3, 3], [1, 1], initializer = initializer, activation = activation, name = 'cnn_conv4')

            # CNN_l5
            self.cnn_l5, self.cnn_w['l5_w'], self.cnn_w['l5_b'] = cov_layer(self.cnn_l4, 256, [3, 3], [1, 1], initializer = initializer, activation = activation, name = 'cnn_conv5')
            self.pool5 = tf.nn.max_pool(self.cnn_l5, ksize = [1, 3, 3, 1], stride = [1, 2, 2, 1], padding = 'SAME', name = 'pool5')

            # CNN_l5 reshape
            self.cnn_l5_flat = tf.reshape(self.pool5, [config.batch_size, -1], name = 'l5_flat')
            
            # CNN_output
            self.cnn_output, self.w['output_w'], self.w['output_b'] = fc_layer(self.cnn_l5_flat, 4096, activation = activation, name = 'output')
            
            return self.cnn_output



    def build_dqn_net(self,config):
        """Build the Deep-Q-Net

        Args:
            config: the global configer which offers all of required configuration.

        """
        # initializer and rectifier
        activation = tf.nn.relu
        initializer = tf.truncated_normal_initializer(config.ini_mean, config.ini_stddev)

        # deep-Q-net training weights(w) and target weights(t_w)
        self.dqn_w = {}
        self.dqn_t_w = {}

        with tf.variable_scope("prediction"):
            # Input_Holder
            inp_size = config.featureDimension + config.actionDimension
            inp = tf.placeholder(config.featureDType, [None, inp_size], 'dqn_input')
            
            self.dqn_l1, self.dqn_w['l1_w'], self.dqn_w['l1_b'] = cov_layer(inp, 32, [8, 8], [4, 4], initializer, activation, name = 'dqn_l1')
            self.dqn_l2, self.dqn_w['l2_w'], self.w['l2_b'] = cov_layer(inp, 64, [4, 4], [2, 2], initializer, activation, name = 'dqn_l2')
            self.dqn_l3, self.dqn_w['l3_w'], self.w['l3_b'] = cov_layer(inp, 64, [3, 3], [1, 1], initializer, activation, name = 'dqn_l3')





    def record(self):

    def train(self):

    def evaluation(self):

