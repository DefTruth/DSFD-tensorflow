#-*-coding:utf-8-*-


# https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/rpn/generate_anchors.py

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import numpy as np
from train_config import config as cfg
# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

# array([[ -83.,  -39.,  100.,   56.],
#       [-175.,  -87.,  192.,  104.],
#       [-359., -183.,  376.,  200.],
#       [ -55.,  -55.,   72.,   72.],
#       [-119., -119.,  136.,  136.],
#       [-247., -247.,  264.,  264.],
#       [ -35.,  -79.,   52.,   96.],
#       [ -79., -167.,   96.,  184.],
#       [-167., -343.,  184.,  360.]])



def generate_anchors(base_size=16,ratios=[0.5,1.,2.],scales=2**np.arange(3,6),rect=cfg.ANCHOR.rect):
    base_anchor=np.array([1,1,base_size,base_size])-1
    anchors_in_ratios=make_anchor_in_ratios(base_anchor,ratios,rect)
    anchors_in_scales=make_anchor_in_sclaes(anchors_in_ratios,scales)

    return anchors_in_scales

def _to_whxy(anchors):
    w=anchors[2]-anchors[0]+1
    h=anchors[3]-anchors[1]+1

    x=anchors[0]+(w-1)/2
    y=anchors[1]+(h-1)/2
    return w,h,x,y

def _to_xyxy(w,h,x,y):

    x0=x-(w-1)/2
    y0=y-(h-1)/2
    x1=x+(w-1)/2
    y1 = y + (h-1) / 2

    return np.stack((x0,y0,x1,y1),axis=-1)

def make_anchor_in_ratios(base_anchor,ratios,rect=False):

    anchors_in_ratios=[]
    w,h,x,y=_to_whxy(base_anchor)
    area=w*h

    for ratio in ratios:
        if rect:
            w=h=np.round(np.sqrt(area/ratio))
            if cfg.ANCHOR.rect_longer:
                h=np.round(1.5*w)
        else:
            w=np.round(np.sqrt(area/ratio))
            h=np.round(ratio*w)

        anchors_in_ratios.append(_to_xyxy(w,h,x,y))


    return np.array(anchors_in_ratios)

def make_anchor_in_sclaes(anchors,scales):
    anchors_res=[]

    for anchor in anchors:
        w,h,x,y=_to_whxy(anchor)
        w=w*scales
        h=h*scales
        anchors_sclase=_to_xyxy(w,h,x,y)
        anchors_res.append(anchors_sclase)
    return np.array(anchors_res).reshape([-1,4])





if __name__=='__main__':
    anchors=generate_anchors()
    print(anchors)
