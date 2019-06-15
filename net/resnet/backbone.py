import tensorflow as tf
import tensorflow.contrib.slim as slim

from train_config import config as cfg




from net.arg_scope.resnet_args_cope import resnet_arg_scope

from net.resnet.resnet_v1 import resnet_v1_50,resnet_v1_101

from net.FEM import create_fem_net


def l2_normalization(x, scale, name):
    with tf.variable_scope(name):
        x = scale*tf.nn.l2_normalize(x, axis=-1)
    return x


def resnet_ssd(image,L2_reg,is_training=True):

    resnet=resnet_v1_101 if '101' in cfg.MODEL.net_structure else resnet_v1_50

    with slim.arg_scope(resnet_arg_scope(weight_decay=L2_reg, bn_is_training=is_training)):
        if cfg.TRAIN.lock_basenet_bn:
            net, end_points  = resnet(image,is_training=False, global_pool=False, num_classes=None)
        else:
            net, end_points  = resnet(image, is_training=is_training, global_pool=False, num_classes=None)

    for k, v in end_points.items():
        print(k, v)

    resnet_fms = [ end_points['resnet_v1_50/block1'],
                   end_points['resnet_v1_50/block2'],
                   end_points['resnet_v1_50/block3'],
                   end_points['resnet_v1_50/block4'],
                   ]

    with slim.arg_scope(resnet_arg_scope(weight_decay=L2_reg, bn_is_training=is_training)):


        net = slim.conv2d(resnet_fms[-1], 256, [1, 1], stride=1, activation_fn=tf.nn.relu,normalizer_fn=None, scope='extra_conv_1_1')
        net = slim.conv2d(net, 512, [3, 3], stride=2, activation_fn=tf.nn.relu,normalizer_fn=None, scope='extra_conv_1_2')
        resnet_fms.append(net)
        net = slim.conv2d(net, 128, [1, 1], stride=1, activation_fn=tf.nn.relu,normalizer_fn=None, scope='extra_conv_2_1')
        net = slim.conv2d(net, 256, [3, 3], stride=2, activation_fn=tf.nn.relu,normalizer_fn=None, scope='extra_conv_2_2')
        resnet_fms.append(net)
        print('extra resnet50 backbone output:', resnet_fms)


        if cfg.MODEL.fpn:
            enhanced_fms = create_fem_net(resnet_fms, L2_reg, is_training)
        else:
            enhanced_fms =None

        enhanced_fms[0] = l2_normalization(enhanced_fms[0], scale=cfg.MODEL.l2_norm[0], name='ef0')
        enhanced_fms[1] = l2_normalization(enhanced_fms[1], scale=cfg.MODEL.l2_norm[1], name='ef1')
        enhanced_fms[2] = l2_normalization(enhanced_fms[2], scale=cfg.MODEL.l2_norm[2], name='ef2')


    return resnet_fms,enhanced_fms
