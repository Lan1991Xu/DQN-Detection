import numpy as np
import tensorflow as tf

def cov_layer(inp,
            output_dim,
            kernel_size,
            stride,
            w_initializer = tf.contrib.layers.xavier_initializer(),
            b_initializer = tf.constant_initializer(0.0),
            activation = tf.nn.relu,
            name = 'conv2d'
            data_format = 'NHWC'
            padding = 'SAME'):
    with tf.variable_scope(name):
        stride = [1, stride[0], stride[1], 1]
        kernel_shape = [kernel_size[0], kernel_size[1], inp.get_shape()[-1], output_dim]

        w = tf.get_variable('w', kernel_shape, tf.float32, initializer = w_initializer)
        conv = tf.nn.conv2d(x, w, stride, padding, data_format = data_format)   
        b = tf.get_variable('biases', [output_dim], initializer = b_initializer)
        out = tf.nn.bias_add(conv, b, data_format)

    if activation != None:
        out = activation(out)

    return out, w, b

def fc_layer(inp,
            output_dim,
            w_initializer = tf.contrib.layers.xavier_initializer(),
            b_initializer = tf.constant_initializer(0.0),
            activation = None,
            bias_start
            name = 'linearfc'):
    shape = inp.get_shape().as_list()
    
    with tf.variable_scope(name):
        w = tf.get_variable('w', [shape[1], output_size], tf.float32, w_initializer))
        b = tf.get_variable('b', [output_dim], initializer = b_initializer)

        out = tf.nn.bias_add(tf.matmul(inp, w), b)

        if activation != None:
            out = activation(out)

    return out, w, b 
