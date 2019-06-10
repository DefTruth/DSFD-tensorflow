import tensorflow as tf
import tensorflow.contrib.slim as slim

from train_config import config as cfg




from net.arg_scope.resnet_args_cope import resnet_arg_scope

from net.resnet.resnet_v2 import resnet_v2_50,resnet_v2_101

from net.FEM import create_fem_net

def resnet_ssd(image,L2_reg,is_training=True):

    resnet=resnet_v2_101 if '101' in cfg.MODEL.net_structure else resnet_v2_50

    with slim.arg_scope(resnet_arg_scope(weight_decay=L2_reg, bn_is_training=is_training)):
        if cfg.TRAIN.lock_basenet_bn:
            net, end_points  = resnet(image,is_training=False, global_pool=False, num_classes=None)
        else:
            net, end_points  = resnet(image, is_training=is_training, global_pool=False, num_classes=None)

    for k, v in end_points.items():
        print(k, v)

    resnet_fms = [ tf.nn.relu(slim.batch_norm(end_points['resnet_v2_50/block1'])),
                   tf.nn.relu(slim.batch_norm(end_points['resnet_v2_50/block2'])),
                   tf.nn.relu(slim.batch_norm(end_points['resnet_v2_50/block3'])),
                   tf.nn.relu(slim.batch_norm(end_points['resnet_v2_50/block4']))]

    with slim.arg_scope(resnet_arg_scope(weight_decay=L2_reg, bn_is_training=is_training)):


        net = slim.conv2d(resnet_fms[-1], 512, [1, 1], stride=1, activation_fn=tf.nn.relu, scope='extra_conv_1_1')
        net = slim.conv2d(net, 512, [3, 3], stride=2, activation_fn=tf.nn.relu, scope='extra_conv_1_2')
        resnet_fms.append(net)
        net = slim.conv2d(net, 128, [1, 1], stride=1, activation_fn=tf.nn.relu, scope='extra_conv_2_1')
        net = slim.conv2d(net, 256, [3, 3], stride=2, activation_fn=tf.nn.relu, scope='extra_conv_2_2')
        resnet_fms.append(net)
        print('extra resnet50 backbone output:', resnet_fms)


        if cfg.MODEL.fpn:
            enhanced_fms = create_fem_net(resnet_fms, L2_reg, is_training)
        else:
            enhanced_fms =None
    return resnet_fms,enhanced_fms
