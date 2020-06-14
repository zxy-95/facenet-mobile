from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import utils
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

def _squeeze_excitation_layer(input, out_dim, ratio):
    squeeze = slim.avg_pool2d(input, input.get_shape()[1:-1], stride=1)
    excitation = slim.convolution2d(squeeze, int(out_dim / ratio), [1, 1], stride=1, activation_fn=relu6)
    excitation = slim.convolution2d(excitation, out_dim, [1, 1], stride=1, activation_fn=hard_sigmoid)
    excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
    scale = input * excitation
    return scale


def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):
  """Define kernel size which is automatically reduced for small input.
  If the shape of the input images is unknown at graph construction time this
  function assumes that the input images are large enough.
  Args:
    input_tensor: input tensor of size [batch_size, height, width, channels].
    kernel_size: desired kernel size of length 2: [kernel_height, kernel_width]
  Returns:
    a tensor with the kernel size.
  """
  shape = input_tensor.get_shape().as_list()
  if shape[1] is None or shape[2] is None:
    kernel_size_out = kernel_size
  else:
    kernel_size_out = [min(shape[1], kernel_size[0]),
                       min(shape[2], kernel_size[1])]
  return kernel_size_out


def _depthwise_separable_conv(inputs,
                              num_pwc_filters,
                              width_multiplier,
                              sc,
                              downsample=False):
    """ Helper function to build the depth-wise separable convolution layer.
    """
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.995,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
    }
    num_pwc_filters = round(num_pwc_filters * width_multiplier)
    _stride = 2 if downsample else 1
    with slim.arg_scope([slim.convolution2d, slim.separable_conv2d], \
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        biases_initializer=tf.zeros_initializer(),
                        # weights_regularizer=slim.l2_regularizer(weight_decay),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        padding='SAME'):
        # skip pointwise by setting num_outputs=None
        depthwise_conv = slim.separable_convolution2d(inputs,
                                                      num_outputs=None,
                                                      stride=_stride,
                                                      depth_multiplier=1,
                                                      kernel_size=[3, 3],
                                                      scope=sc + '/depthwise_conv')



        pointwise_conv = slim.convolution2d(depthwise_conv,
                                            num_pwc_filters,
                                            kernel_size=[1, 1],
                                            scope=sc + '/pointwise_conv')

    return pointwise_conv

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
    print("improved mobilenet")
    width_multiplier=0.5
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=slim.xavier_initializer_conv2d(uniform=True),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        with tf.variable_scope('mobilenet', [images], reuse=reuse):
            with slim.arg_scope([slim.batch_norm, slim.dropout],
                                is_training=phase_train):
                net = slim.convolution2d(images, round(32 * width_multiplier), [3, 3], stride=2, padding='SAME', scope='conv_1')
                net = _depthwise_separable_conv(net, 64, width_multiplier, sc='conv_ds_2')
                net = _depthwise_separable_conv(net, 128, width_multiplier, downsample=True, sc='conv_ds_3')
                net = _squeeze_excitation_layer(net, int(128*width_multiplier), 8)

                net = _depthwise_separable_conv(net, 128, width_multiplier, sc='conv_ds_4')
                net1 = slim.convolution2d(net, int(128 * width_multiplier), kernel_size=[1, 1], scope='conv1')
                net2 = _depthwise_separable_conv(net, 128, width_multiplier, sc='conv_ds_4_1')
                net3 = _depthwise_separable_conv(net2, 128, width_multiplier, sc='conv_ds_4_2')
                net = tf.add_n([net1,net3])
                net = tf.nn.relu6(net)
                net = _squeeze_excitation_layer(net, int(128 * width_multiplier), 8)
                net = _depthwise_separable_conv(net, 256, width_multiplier, downsample=True, sc='conv_ds_5')
                net = _squeeze_excitation_layer(net, int(256 * width_multiplier), 8)
                net = _depthwise_separable_conv(net, 256, width_multiplier, sc='conv_ds_6')
                net = _squeeze_excitation_layer(net, int(256 * width_multiplier), 8)
                net = _depthwise_separable_conv(net, 384, width_multiplier, downsample=True, sc='conv_ds_7')
                net_512_1 = _squeeze_excitation_layer(net, int(384 * width_multiplier), 8)


                net_1_1=slim.convolution2d(net_512_1, round(128 * width_multiplier), [1, 1], padding='SAME')
                net_1_1 = slim.convolution2d(net_1_1, round(384 * width_multiplier), [1, 1], padding='SAME',activation_fn=None)

                net_1_2 =slim.convolution2d(net_512_1, round(128 * width_multiplier), [1, 1], padding='SAME')
                net_1_2 = _depthwise_separable_conv(net_1_2, 128, width_multiplier, sc='conv_ds_8')
                net_1_2 = _squeeze_excitation_layer(net_1_2, int(128 * width_multiplier), 8)
                net_1_2= slim.convolution2d(net_1_2, round(384 * width_multiplier), [1, 1], padding='SAME',activation_fn=None)
                net_1_3=tf.add_n([net_1_1,net_1_2])
                net=tf.add_n([net_512_1+net_1_3])
                net_512_1=tf.nn.relu6(net)

                net_1_1 = slim.convolution2d(net_512_1, round(128 * width_multiplier), [1, 1], padding='SAME')
                net_1_1 = slim.convolution2d(net_1_1, round(384 * width_multiplier), [1, 1], padding='SAME',
                                             activation_fn=None)

                net_1_2 = slim.convolution2d(net_512_1, round(128 * width_multiplier), [1, 1], padding='SAME')
                net_1_2 = _depthwise_separable_conv(net_1_2, 128, width_multiplier, sc='conv_ds_9')
                net_1_2 = _squeeze_excitation_layer(net_1_2, int(128 * width_multiplier), 8)
                net_1_2 = slim.convolution2d(net_1_2, round(384 * width_multiplier), [1, 1], padding='SAME',
                                             activation_fn=None)
                net_1_3 = tf.add_n([net_1_1, net_1_2])
                net = tf.add_n([net_512_1 + net_1_3])
                net_512_1 = tf.nn.relu6(net)

                net_1_1 = slim.convolution2d(net_512_1, round(128 * width_multiplier), [1, 1], padding='SAME')
                net_1_1 = slim.convolution2d(net_1_1, round(384 * width_multiplier), [1, 1], padding='SAME',
                                             activation_fn=None)

                net_1_2 = slim.convolution2d(net_512_1, round(128 * width_multiplier), [1, 1], padding='SAME')
                net_1_2 = _depthwise_separable_conv(net_1_2, 128, width_multiplier, sc='conv_ds_10')
                net_1_2 = _squeeze_excitation_layer(net_1_2, int(128 * width_multiplier), 8)
                net_1_2 = slim.convolution2d(net_1_2, round(384 * width_multiplier), [1, 1], padding='SAME',
                                             activation_fn=None)
                net_1_3 = tf.add_n([net_1_1, net_1_2])
                net = tf.add_n([net_512_1 + net_1_3])
                net_512_1 = tf.nn.relu6(net)

                net_1_1 = slim.convolution2d(net_512_1, round(128 * width_multiplier), [1, 1], padding='SAME')
                net_1_1 = slim.convolution2d(net_1_1, round(384 * width_multiplier), [1, 1], padding='SAME',
                                             activation_fn=None)

                net_1_2 = slim.convolution2d(net_512_1, round(128 * width_multiplier), [1, 1], padding='SAME')
                net_1_2 = _depthwise_separable_conv(net_1_2, 128, width_multiplier, sc='conv_ds_11')
                net_1_2 = _squeeze_excitation_layer(net_1_2, int(128 * width_multiplier), 8)
                net_1_2 = slim.convolution2d(net_1_2, round(384 * width_multiplier), [1, 1], padding='SAME',
                                             activation_fn=None)
                net_1_3 = tf.add_n([net_1_1, net_1_2])
                net = tf.add_n([net_512_1 + net_1_3])
                net = tf.nn.relu6(net)


                net = _depthwise_separable_conv(net, 768, width_multiplier, downsample=True, sc='conv_ds_13')
                net = _squeeze_excitation_layer(net, int(768 * width_multiplier), 8)
                net = _depthwise_separable_conv(net, 1024, width_multiplier, sc='conv_ds_14')
                net = _squeeze_excitation_layer(net, int(1024 * width_multiplier), 8)
                # net = slim.avg_pool2d(net, [7, 7], scope='avg_pool_15')
                # net = tf.reduce_mean(net, [1, 2], name='avg_pool_15', keep_dims=True)
                kernel_size = _reduced_kernel_size_for_small_input(net, [4, 4])

                # Global depthwise conv2d
                net = slim.separable_conv2d(inputs=net, num_outputs=None, kernel_size=kernel_size, stride=1,
                                            depth_multiplier=1.0, activation_fn=None, padding='VALID')
                net = slim.conv2d(inputs=net, num_outputs=int(1024 * width_multiplier), kernel_size=[1, 1], stride=1, activation_fn=None,
                                  padding='VALID')

                net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
                net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None, scope='fc_16')

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
    inputs = tf.placeholder(dtype=tf.float32, shape=[1,128,128,3])
    inference(inputs, 0,bottleneck_layer_size=512)
