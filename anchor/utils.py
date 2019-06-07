#-*-coding:utf-8-*-

import sys
sys.path.append('.')

import numpy as np
from train_config import config as cfg


from anchor.base_anchor import generate_anchors
from anchor.common import np_iou,filter_boxes_inside_shape
from anchor.box_utils import encode


def produce_target(image,boxes,labels):
    boxes = boxes.copy()



    all_anchors_flatten = cfg.ANCHOR.achors


    #inside_ind, inside_anchors = filter_boxes_inside_shape(all_anchors_flatten, image.shape[:2])
    inside_anchors=all_anchors_flatten

    # obtain anchor labels and their corresponding gt boxes
    anchor_labels, anchor_gt_boxes = get_anchor_labels(inside_anchors, boxes,labels)

    # map back to all_anchors
    num_all_anchors = all_anchors_flatten.shape[0]
    all_labels = np.zeros((num_all_anchors, ), dtype='int32')
    #all_labels[inside_ind] = anchor_labels
    all_labels=anchor_labels
    all_boxes = np.zeros((num_all_anchors, 4), dtype='float32')
    #all_boxes[inside_ind] = anchor_gt_boxes
    all_boxes=anchor_gt_boxes

    if boxes.shape[0]==0:
        print('hihihi')
        all_labels = np.zeros((num_all_anchors,), dtype='int32')
        all_boxes = np.zeros((num_all_anchors, 4), dtype='float32')

    # start = 0
    # multilevel_inputs = []
    # for level_anchor in anchors_per_level:
    #     assert level_anchor.shape[2] == len(cfg.ANCHOR.ANCHOR_RATIOS)
    #     anchor_shape = level_anchor.shape[:3]   # fHxfWxNUM_ANCHOR_RATIOS
    #     num_anchor_this_level = np.prod(anchor_shape)
    #     end = start + num_anchor_this_level
    #     multilevel_inputs.append(
    #         (all_labels[start: end].reshape(anchor_shape),
    #          all_boxes[start: end, :].reshape(anchor_shape + (4,))
    #          ))
    #     start = end
    # assert end == num_all_anchors, "{} != {}".format(end, num_all_anchors)
    # return multilevel_inputs
    return all_boxes,all_labels

def get_anchor_labels(anchors, gt_boxes,labels):
    # This function will modify labels and return the filtered inds
    def filter_box_label(labels, value, max_num):
        curr_inds = np.where(labels == value)[0]
        if len(curr_inds) > max_num:
            disable_inds = np.random.choice(
                curr_inds, size=(len(curr_inds) - max_num),
                replace=False)
            labels[disable_inds] = -1  # ignore them
            curr_inds = np.where(labels == value)[0]
        return curr_inds

    NA, NB = len(anchors), len(gt_boxes)
    assert NB > 0  # empty images should have been filtered already
    # ##########
    anchor_matched_already = np.zeros((NA,), dtype='int32')
    gt_boxes_mathed_already = np.zeros((NB,), dtype='int32')
    anchor_labels=np.zeros((NA,), dtype='int32')
    anchor_boxes = np.zeros((NA, 4), dtype='float32')

    box_ious = np_iou(anchors, gt_boxes)  # NA x NB


    # for each anchor box choose the groundtruth box with largest iou
    max_iou = box_ious.max(axis=1)  # NA
    positive_anchor_indices = np.where(max_iou > cfg.ANCHOR.POSITIVE_ANCHOR_THRESH)[0]
    negative_anchor_indices = np.where(max_iou < cfg.ANCHOR.NEGATIVE_ANCHOR_THRESH)[0]

    positive_iou = box_ious[positive_anchor_indices]
    matched_gt_box_indices = positive_iou.argmax(axis=1)

    anchor_labels[positive_anchor_indices]=labels[matched_gt_box_indices]
    anchor_boxes[positive_anchor_indices]=gt_boxes[matched_gt_box_indices]
    anchor_matched_already[positive_anchor_indices]=1#### marked as matched
    gt_boxes_mathed_already[matched_gt_box_indices]=1#### marked as matched


    if np.sum(anchor_matched_already)>0:
        n=np.sum(anchor_matched_already)/np.sum(gt_boxes_mathed_already)
    else:
        n=cfg.ANCHOR.AVG_MATCHES
    n= n if n >cfg.ANCHOR.AVG_MATCHES else cfg.ANCHOR.AVG_MATCHES
    if not cfg.ANCHOR.super_match:
        n=cfg.ANCHOR.AVG_MATCHES
    # some gt_boxes may not matched, find them and match them with n anchors for each gt box
    box_ious[box_ious<cfg.ANCHOR.NEGATIVE_ANCHOR_THRESH]=0
    sorted_ious=np.argsort(-box_ious,axis=0)

    sorted_ious=sorted_ious[np.logical_not(anchor_matched_already)]

    for i in range(0,len(gt_boxes_mathed_already)):
        matched_count=np.sum(matched_gt_box_indices==gt_boxes_mathed_already[i])

        if matched_count>=n:
            continue
        else:
            for j in range(0,int(n-matched_count)):
                if box_ious[sorted_ious[j][i]][i]>cfg.ANCHOR.NEGATIVE_ANCHOR_THRESH:
                    anchor_labels[sorted_ious[j][i]]= labels[i]
                    anchor_boxes[sorted_ious[j][i]] = gt_boxes[i]

                    anchor_matched_already[sorted_ious[j][i]]=1

                    gt_boxes_mathed_already[i]=1

    fg_boxes=anchor_boxes[anchor_matched_already.astype(np.bool)]

    matched_anchors=anchors[anchor_matched_already.astype(np.bool)]



    ##select and normlised
    fg_boxes=fg_boxes/cfg.DATA.MAX_SIZE
    matched_anchors=matched_anchors/cfg.DATA.MAX_SIZE
    encode_fg_boxes=encode(fg_boxes,matched_anchors)
    anchor_boxes[anchor_matched_already.astype(np.bool)] = encode_fg_boxes
    # assert len(fg_inds) + np.sum(anchor_labels == 0) == cfg.ANCHOR.BATCH_PER_IM
    return anchor_labels, anchor_boxes



def get_all_anchors(stride=None, sizes=None):
    """
    Get all anchors in the largest possible image, shifted, floatbox
    Args:
        stride (int): the stride of anchors.
        sizes (tuple[int]): the sizes (sqrt area) of anchors

    Returns:
        anchors: SxSxNUM_ANCHORx4, where S == ceil(MAX_SIZE/STRIDE), floatbox
        The layout in the NUM_ANCHOR dim is NUM_RATIO x NUM_SIZE.

    """
    if stride is None:
        stride = cfg.ANCHOR.ANCHOR_STRIDE
    if sizes is None:
        sizes = cfg.ANCHOR.ANCHOR_SIZES
    # Generates a NAx4 matrix of anchor boxes in (x1, y1, x2, y2) format. Anchors
    # are centered on stride / 2, have (approximate) sqrt areas of the specified
    # sizes, and aspect ratios as given.
    cell_anchors = generate_anchors(
        stride,
        scales=np.array(sizes, dtype=np.float) / stride,
        ratios=np.array(cfg.ANCHOR.ANCHOR_RATIOS, dtype=np.float))
    # anchors are intbox here.
    # anchors at featuremap [0,0] are centered at fpcoor (8,8) (half of stride)

    max_size = cfg.DATA.MAX_SIZE
    field_size = int(np.ceil(max_size / stride))
    shifts = np.arange(0, field_size) * stride
    shift_x, shift_y = np.meshgrid(shifts, shifts)
    shift_x = shift_x.flatten()
    shift_y = shift_y.flatten()
    shifts = np.vstack((shift_x, shift_y, shift_x, shift_y)).transpose()
    # Kx4, K = field_size * field_size
    K = shifts.shape[0]

    A = cell_anchors.shape[0]
    field_of_anchors = (
        cell_anchors.reshape((1, A, 4)) +
        shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    field_of_anchors = field_of_anchors.reshape((field_size, field_size, A, 4))
    # FSxFSxAx4
    # Many rounding happens inside the anchor code anyway
    # assert np.all(field_of_anchors == field_of_anchors.astype('int32'))
    field_of_anchors = field_of_anchors.astype('float32')
    field_of_anchors[:, :, :, [2, 3]] += 1
    return field_of_anchors

def get_all_anchors_fpn(strides=None, sizes=None):
    """
    Returns:
        [anchors]: each anchors is a SxSx NUM_ANCHOR_RATIOS x4 array.
    """
    if strides is None:
        strides = cfg.ANCHOR.ANCHOR_STRIDES
    if sizes is None:
        sizes = cfg.ANCHOR.ANCHOR_SIZES
    assert len(strides) == len(sizes)
    foas = []
    for stride, size in zip(strides, sizes):
        foa = get_all_anchors(stride=stride, sizes=(size,))
        foas.append(foa)
    return foas


if __name__ == '__main__':

    import cv2
    anchors=cfg.ANCHOR.achors

    image=np.ones(shape=[cfg.DATA.MAX_SIZE,cfg.DATA.MAX_SIZE,3])*255

    # for x in anchors:
    #     print(x.shape)

    anchors=np.array(anchors)

    for i in range(0,anchors.shape[0]):
        box=anchors[i]
        print(box[2]-box[0])
        cv2.rectangle(image, (int(box[0]), int(box[1])),
                      (int(box[2]), int(box[3])), (255, 0, 0), 1)

        cv2.namedWindow('anchors',0)
        cv2.imshow('anchors',image)
        cv2.waitKey(0)

    a,b=produce_target(image,np.array([[34., 396.,  58., 508.],[20,140,50,160]]),np.array([1,1]))

    # print(a.shape)
    # print(b.shape)
    #
    #
    #
    # for i in range(len(a)):
    #     label_target=b[i]
    #     boxes_target=a[i]
    #
    #
    #
    #     if label_target>0:
    #         box=boxes_target
    #         print(box)

