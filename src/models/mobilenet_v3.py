from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import keras.backend as K

def relu6(x, name='relu6'):
    return tf.nn.relu6(x, name)
def hard_swish(x, name='hard_swish'):
    with tf.variable_scope(name):
        h_swish = x * tf.nn.relu6(x + 3) / 6
    return h_swish
def hard_sigmoid(x, name='hard_sigmoid'):
    with tf.variable_scope(name):
        h_sigmoid = tf.nn.relu6(x + 3) / 6
    return h_sigmoid
def _squeeze_excitation_layer(input, out_dim, ratio,is_training=True, reuse=None):
    squeeze = slim.avg_pool2d(input,input.get_shape()[1:-1], stride=1)
    excitation=slim.convolution2d(squeeze,int(out_dim / ratio),[1,1],stride=1,activation_fn=relu6)
    excitation=slim.convolution2d(excitation,out_dim,[1,1],stride=1,activation_fn=hard_sigmoid)
    excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
    scale = input * excitation
    return scale

def mobilenet_v3_block(input, kernel, batch_norm_params,expansion_dim, output_dim, stride, name, is_training=True,
                       shortcut=True, activatation="RE", ratio=16, se=False):
    if activatation == "HS":
        activation_fn= hard_swish
    elif activatation == "RE":
        activation_fn= relu6
    with tf.variable_scope(name):
        with slim.arg_scope([slim.convolution2d, slim.separable_conv2d], \
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            biases_initializer=tf.zeros_initializer(),
                            #weights_regularizer=slim.l2_regularizer(weight_decay),
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            padding='SAME'):
            net=slim.convolution2d(input,expansion_dim,[1,1],stride=1,activation_fn=activation_fn)
            net=slim.separable_convolution2d(net,num_outputs=None, kernel_size=kernel,depth_multiplier=1,stride=stride,activation_fn=activation_fn)
            if se:
                channel = net.get_shape().as_list()[-1]
                net = _squeeze_excitation_layer(net, out_dim=channel, ratio=ratio)
            net=slim.convolution2d(net,output_dim,[1,1],stride=1,activation_fn=None)
            if shortcut and stride == 1:
                net += input
            return net


def inference(images, keep_probability, phase_train=True, bottleneck_layer_size=128, weight_decay=0.0, reuse=None):
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.995,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
    }
    layers = [
        [16, 16, 3, 2, "RE", True, 16],
        [16, 24, 3, 2, "RE", False, 72],
        [24, 24, 3, 1, "RE", False, 88],
        [24, 40, 5, 2, "RE", True, 96],
        [40, 40, 5, 1, "RE", True, 240],
        [40, 40, 5, 1, "RE", True, 240],
        [40, 48, 5, 1, "HS", True, 120],
        [48, 48, 5, 1, "HS", True, 144],
        [48, 96, 5, 2, "HS", True, 288],
        [96, 96, 5, 1, "HS", True, 576],
        [96, 96, 5, 1, "HS", True, 576],
    ]
    multiplier=1
    end_points = {}
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=slim.xavier_initializer_conv2d(uniform=True),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        with tf.variable_scope('squeezenet', [images], reuse=reuse):
            with slim.arg_scope([slim.batch_norm, slim.dropout],
                                is_training=phase_train):
                x = slim.convolution2d(images, int(16 * multiplier), [3, 3], stride=2, activation_fn=hard_swish)
                for idx, (in_channels, out_channels, kernel_size, stride, activatation, se, exp_size) in enumerate(
                        layers):
                    in_channels = int(in_channels * multiplier)
                    out_channels = int(out_channels * multiplier)
                    exp_size = int(exp_size * multiplier)
                    x = mobilenet_v3_block(x, [kernel_size, kernel_size], batch_norm_params, exp_size, out_channels,
                                           stride,
                                           "bneck{}".format(idx),shortcut=(in_channels == out_channels), activatation=activatation,se=se)
                    end_points["bneck{}".format(idx)] = x
                x = slim.convolution2d(x, int(576 * multiplier), [1, 1], stride=1)
                net = slim.avg_pool2d(x, x.get_shape()[1:3], scope='avgpool10')
                net = tf.squeeze(net, [1, 2], name='logits')
                net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None,
                        scope='Bottleneck', reuse=False)
                sess = K.get_session()
                graph = sess.graph
                stats_graph(graph)

    return net, None


def stats_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    inputs = tf.placeholder(dtype=tf.float32, shape=[1, 128,128,3])
    inference(inputs,0.8,bottleneck_layer_size=512)