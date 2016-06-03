import numpy as np
import tensorflow as tf

def cov_layer(inp,
            output_dim,
            kernel_size,
            stride,
            initializer = tf.contrib.layers.xavier_initializer(),
            activation = tf.nn.relu,
            name = 'conv2d'):

