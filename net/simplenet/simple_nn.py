#-*-coding:utf-8-*-

import tensorflow as tf
import tensorflow.contrib.slim as slim
from train_config import config as cfg

from net.GN import GroupNorm

def shufflenet_arg_scope(weight_decay=0.00001,
                     batch_norm_decay=0.99,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True,
                     use_batch_norm=True,
                     is_training=True,
                     trainable=True,
                     batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS,
                     ):
  """Defines the default ResNet arg scope.
  TODO(gpapan): The batch-normalization related default values above are
    appropriate for use in conjunction with the reference ResNet models
    released at https://github.com/KaimingHe/deep-residual-networks. When
    training ResNets from scratch, they might need to be tuned.
  Args:
    weight_decay: The weight decay to use for regularizing the model.
    batch_norm_decay: The moving average decay when estimating layer activation
      statistics in batch normalization.
    batch_norm_epsilon: Small constant to prevent division by zero when
      normalizing activations by their variance in batch normalization.
    batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
      activations in the batch normalization layer.
    activation_fn: The activation function which is used in ResNet.
    use_batch_norm: Whether or not to use batch normalization.
    batch_norm_updates_collections: Collection for the update ops for
      batch norm.
  Returns:
    An `arg_scope` to use for the resnet models.
  """
  batch_norm_params = {
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'scale': batch_norm_scale,
      'updates_collections': batch_norm_updates_collections,
      'fused': True,  # Use fused batch norm if possible.
      'trainable':True,
      'is_training':is_training
  }

  with slim.arg_scope(
      [slim.conv2d,slim.separable_conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=slim.variance_scaling_initializer(),
      normalizer_fn=slim.batch_norm if 'BN' in cfg.TRAIN.norm else GroupNorm,
      normalizer_params=batch_norm_params if 'BN' in cfg.TRAIN.norm else None,
      biases_initializer=None):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      # The following implies padding='SAME' for pool1, which makes feature
      # alignment easier for dense prediction tasks. This is also used in
      # https://github.com/facebook/fb.resnet.torch. However the accompanying
      # code of 'Deep Residual Learning for Image Recognition' uses
      # padding='VALID' for pool1. You can switch to that choice by setting
      # slim.arg_scope([slim.max_pool2d], padding='VALID').
      with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
        return arg_sc



######channel shuffle unit
def concat_shuffle_split(x, y):
    with tf.name_scope('concat_shuffle_split'):
        shape = tf.shape(x)
        batch_size = shape[0]
        height, width = shape[1], shape[2]

        depth = x.shape[3].value

        z = tf.stack([x, y], axis=3)  # shape [batch_size, height, width, 2, depth]

        z = tf.transpose(z, [0, 1, 2, 4, 3])
        z = tf.reshape(z, [batch_size, height, width, 2*depth])
        x, y = tf.split(z, num_or_size_splits=2, axis=3)
        return x, y


#### a depwise conv unit
def basic_unit(x):
    in_channels = x.shape[3].value

    x = slim.conv2d(x, in_channels, [1, 1], stride=1, activation_fn=tf.nn.relu, scope='conv1x1_before')

    x = slim.separable_conv2d(x, in_channels, [3, 3], stride=1, activation_fn=None, scope='depthwise', depth_multiplier=1)

    x = slim.conv2d(x, in_channels, [1, 1], stride=1, activation_fn=tf.nn.relu, scope='conv1x1_after')
    return x
#### a depwise conv unit with downsampling
def basic_unit_with_downsampling(x,out_channels=None):
    in_channels = x.shape[3].value
    out_channels = 2 * in_channels if out_channels is None else out_channels

    y = slim.conv2d(x, in_channels, [1, 1], stride=1, activation_fn=tf.nn.relu, scope='conv1x1_before')

    y = slim.separable_conv2d(y, in_channels, [3, 3], stride=2, activation_fn=None, scope='depthwise',depth_multiplier=1)

    y = slim.conv2d(y, out_channels//2, [1, 1], stride=1, activation_fn=tf.nn.relu, scope='conv1x1_after')


    with tf.variable_scope('second_branch'):
        x = slim.separable_conv2d(x, in_channels, [3, 3], stride=2, activation_fn=None, scope='depthwise',depth_multiplier=1)
        x = slim.conv2d(x, out_channels // 2, [1, 1], stride=1, activation_fn=tf.nn.relu, scope='conv1x1_after')


    return x,y

### a simple conv unit
def basic_unit_plain(x):
    in_channels = x.shape[3].value

    x1 = slim.conv2d(x, in_channels//2, [3, 3], stride=1, activation_fn=tf.nn.relu, scope='conv1x1_before1')

    x2 = slim.conv2d(x, in_channels//2, [3, 3], stride=1, activation_fn=tf.nn.relu, scope='conv1x1_before2')
    x2 = slim.conv2d(x2, in_channels//2, [3, 3], stride=1, activation_fn=tf.nn.relu, scope='conv1x1_before3')

    x = tf.concat([x1, x2], axis=3)
    return x

### a simple conv unit with downsampling
def basic_unit_with_downsampling_plain(x,out_channels=None,decrese=True):

    stride =2 if decrese else 1
    in_channels = x.shape[3].value
    out_channels = 2 * in_channels if out_channels is None else out_channels

    y = slim.conv2d(x, out_channels//2, [3, 3], stride=stride, activation_fn=tf.nn.relu, scope='conv1x1_before')

    with tf.variable_scope('second_branch'):

        if decrese:
            x = slim.conv2d(x, out_channels // 2, [3, 3], stride=stride, activation_fn=tf.nn.relu, scope='conv1x1_after1')
        x = slim.conv2d(x, out_channels // 2, [3, 3], stride=1, activation_fn=tf.nn.relu, scope='conv1x1_after2')

    x = tf.concat([x, y], axis=3)
    return x


def block_plain(x, num_units,out_channels=None,decrese=True, scope='stage'):
    with tf.variable_scope(scope):

        with tf.variable_scope('unit_1'):
            x = basic_unit_with_downsampling_plain(x,out_channels,decrese)

        for j in range(2, num_units + 1):
            with tf.variable_scope('unit_%d' % j):

                x = basic_unit_plain(x)

    return x

def block(x, num_units,out_channels=None, scope='stage'):
    with tf.variable_scope(scope):

        with tf.variable_scope('unit_1'):
            x = basic_unit_with_downsampling(x,out_channels)

        for j in range(2, num_units + 1):
            with tf.variable_scope('unit_%d' % j):
                x = basic_unit(x)

    return x

def block_with_shuffle(x, num_units,out_channels=None, scope='stage'):
    with tf.variable_scope(scope):

        with tf.variable_scope('unit_1'):
            x, y = basic_unit_with_downsampling(x,out_channels)

        for j in range(2, num_units + 1):
            with tf.variable_scope('unit_%d' % j):
                x, y = concat_shuffle_split(x, y)

                x = basic_unit(x)

        x = tf.concat([x, y], axis=3)

    return x

def block_plain_with_shuffle(x, num_units,out_channels=None, scope='stage'):
    with tf.variable_scope(scope):

        with tf.variable_scope('unit_1'):
            x, y = basic_unit_with_downsampling_plain(x,out_channels)

        for j in range(2, num_units + 1):
            with tf.variable_scope('unit_%d' % j):
                x, y = concat_shuffle_split(x, y)

                x = basic_unit_plain(x)

        x = tf.concat([x, y], axis=3)

    return x



def simple_nn(inputs,L2_reg,training=True):

    fms=[]

    arg_scope = shufflenet_arg_scope(weight_decay=L2_reg,is_training=training,)
    with slim.arg_scope(arg_scope):

        with tf.variable_scope('simpleface'):

            net = slim.conv2d(inputs, 32, [3, 3],stride=1, activation_fn=tf.nn.relu, scope='init_conv0')
            net = slim.conv2d(net, 32, [3, 3], stride=2, activation_fn=tf.nn.relu, scope='init_conv1')
            #net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name='pool1')
            net = slim.conv2d(net, 64, [3, 3], stride=1, activation_fn=tf.nn.relu, scope='init_conv2_1')
            net = slim.conv2d(net, 128, [3, 3], stride=2, activation_fn=tf.nn.relu, scope='init_conv2_2')
            #net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name='pool2')
            #fms.append(net)

            print('first conv shape', net.shape)
            net = block_plain(net, num_units=2, out_channels=256, scope='Stage2')
            print('2 conGv shape', net.shape)
            fms.append(net)
            net = block_plain(net, num_units=2, out_channels=256, scope='Stage3')
            print('3 conv shape', net.shape)
            fms.append(net)
            net = block_plain(net, num_units=2, out_channels=512, scope='Stage4')
            fms.append(net)
            print('4 conv shape', net.shape)
            net = block_plain(net, num_units=2, out_channels=512, scope='Stage5')
            fms.append(net)
            print('5 conv shape', net.shape)
            net = block_plain(net, num_units=2, out_channels=512, scope='Stage6')
            fms.append(net)
            print('6 conv shape', net.shape)
            ##use three layes feature in different scale 28x28 14x14 7x7


    return fms


