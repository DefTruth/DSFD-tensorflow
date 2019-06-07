#-*-coding:utf-8-*-

import sys
sys.path.append('.')
import numpy as np
import cv2
import random
import copy
import traceback

from anchor.utils import produce_target


from helper.logger import logger
from data.datainfo import data_info
from data.augmentor.augmentation import Pixel_jitter,Swap_change_aug,Random_contrast,Random_saturation,\
    Random_brightness,Random_scale_withbbox,Random_flip,Blur_aug,Rotate_with_box,Gray_aug,Fill_img,baidu_aug,dsfd_aug

from train_config import config as cfg

from tensorpack.dataflow import DataFromList



def balance(anns):

    def area(boxes):
        """Computes area of boxes.

        Arguments:
            boxes: a float tensor with shape [N, 4].
        Returns:
            a float tensor with shape [N] representing box areas.
        """

        ymin, xmin, ymax, xmax = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        return (ymax - ymin) * (xmax - xmin)

    res_anns=copy.deepcopy(anns)


    small_face=0
    media_face =0
    large_face=0
    for ann in anns:
        label = ann[-1]
        boxes = label.split(' ')
        for box in boxes:
            box=np.array([box.split(',')], dtype=np.float)

            face_area=area(box)
            if face_area<=100:
                small_face+=1
                for i in range(0):
                    res_anns.append(ann)
                break
            elif face_area>100 and face_area<=1600:
                media_face+=1
                for i in range(0):
                    res_anns.append(ann)
                break

            else:
                large_face += 1
                break
    logger.info('befor balance the dataset contains %d smallfaces %d mediafaces %d largefaces' % (small_face,media_face,large_face))

    small_face = 0
    media_face = 0
    large_face = 0
    for ann in res_anns:
        label = ann[-1]
        boxes = label.split(' ')
        for box in boxes:
            box = np.array([box.split(',')], dtype=np.float)

            face_area = area(box)
            if face_area <= 100:
                small_face += 1
                break
            elif face_area > 100 and face_area <= 1600:
                media_face += 1
                break
            else:
                large_face += 1
                break

    logger.info('after balance the dataset contains %d smallfaces %d mediafaces %d largefaces' % (
    small_face, media_face, large_face))

    ##now the balance done nothing
    random.shuffle(res_anns)
    logger.info('befor balance the dataset contains %d images' % (len(anns)))
    logger.info('after balanced the datasets contains %d samples' % (len(res_anns)))
    return res_anns
def get_train_data_list(im_root_path, ann_txt):
    """
    train_im_path : image folder name
    train_ann_path : coco json file name
    """
    logger.info("[x] Get data from {}".format(im_root_path))
    # data = PoseInfo(im_path, ann_path, False)
    data = data_info(im_root_path, ann_txt)
    all_samples=data.get_all_sample()

    return all_samples
def get_data_set(root_path,ana_path):
    data_list=get_train_data_list(root_path,ana_path)
    data_list=balance(data_list)
    dataset= DataFromList(data_list, shuffle=True)
    return dataset

def _data_aug_fn(fname, ground_truth,is_training=True):
    """Data augmentation function."""
    ####customed here
    try:

        image = cv2.imread(fname, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        labels = ground_truth.split(' ')
        boxes = []
        for label in labels:
            bbox = np.array(label.split(','), dtype=np.float)
            ##the augmentor need ymin,xmin,ymax,xmax
            boxes.append([bbox[0], bbox[1], bbox[2], bbox[3],bbox[4]])

        boxes = np.array(boxes, dtype=np.float)

        ###clip the bbox for the reason that some bboxs are beyond the image
        h_raw_limit, w_raw_limit, _ = image.shape
        boxes[:, 2] = np.clip(boxes[:, 2], 0, w_raw_limit)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, h_raw_limit)
        boxes[boxes < 0] = 0
        #########random scale
        ############## becareful with this func because there is a Infinite loop in its body
        if is_training:
            if random.uniform(0, 1) > 0.5:
                image, boxes = Random_flip(image, boxes)
            image = Pixel_jitter(image, max_=15)
            if random.uniform(0, 1) > 0.5:
                image = Random_brightness(image, 35)
            if random.uniform(0, 1) > 0.5:
                image = Random_contrast(image, [0.5, 1.5])
            if random.uniform(0, 1) > 0.5:
                image = Random_saturation(image, [0.5, 1.5])
            if random.uniform(0, 1) > 0.5:
                a = [3, 5, 7, 9]
                k = random.sample(a, 1)[0]
                image = Blur_aug(image, ksize=(k, k))
            if random.uniform(0, 1) > 0.7:
                image = Gray_aug(image)
            # if random.uniform(0, 1) > 0.7:
            #     image = Swap_change_aug(image)
            # if random.uniform(0, 1) > 0.7:
            #     boxes_ = boxes[:, 0:4]
            #     klass_ = boxes[:, 4:]
            #     angle = random.sample([-90, 90], 1)[0]
            #     image, boxes_ = Rotate_with_box(image, boxes=boxes_, angle=angle)
            #     boxes = np.concatenate([boxes_, klass_], axis=1)

            sample_dice=random.uniform(0, 1)
            if  sample_dice> 1:
                if not cfg.DATA.MUTISCALE:
                    image, boxes = Random_scale_withbbox(image, boxes, target_shape=[cfg.DATA.hin, cfg.DATA.win],
                                                         jitter=0.3)
                else:
                    rand_h = random.sample(cfg.DATA.scales, 1)[0]
                    rand_w = random.sample(cfg.DATA.scales, 1)[0]
                    image, boxes = Random_scale_withbbox(image, boxes, target_shape=[rand_h, rand_w], jitter=0.3)
            elif sample_dice>0.5 and sample_dice<=1:
                boxes_ = boxes[:, 0:4]
                klass_ = boxes[:, 4:]

                image, boxes_, klass_ = dsfd_aug(image, boxes_, klass_)
                image, shift_x, shift_y = Fill_img(image, target_width=cfg.DATA.win, target_height=cfg.DATA.hin)
                boxes_[:, 0:4] = boxes_[:, 0:4] + np.array([shift_x, shift_y, shift_x, shift_y], dtype='float32')
                h, w, _ = image.shape
                boxes_[:, 0] /= w
                boxes_[:, 1] /= h
                boxes_[:, 2] /= w
                boxes_[:, 3] /= h
                image = image.astype(np.uint8)
                image = cv2.resize(image, (cfg.DATA.win, cfg.DATA.hin))

                boxes_[:, 0] *= cfg.DATA.win
                boxes_[:, 1] *= cfg.DATA.hin
                boxes_[:, 2] *= cfg.DATA.win
                boxes_[:, 3] *= cfg.DATA.hin
                image = image.astype(np.uint8)
                boxes = np.concatenate([boxes_, klass_], axis=1)
            else:
                boxes_ = boxes[:, 0:4]
                klass_ = boxes[:, 4:]
                image,boxes_,klass_=baidu_aug(image,boxes_,klass_)

                image, shift_x, shift_y = Fill_img(image, target_width=cfg.DATA.win, target_height=cfg.DATA.hin)
                boxes_[:, 0:4] = boxes_[:, 0:4] + np.array([shift_x, shift_y, shift_x, shift_y], dtype='float32')
                h, w, _ = image.shape
                boxes_[:, 0] /= w
                boxes_[:, 1] /= h
                boxes_[:, 2] /= w
                boxes_[:, 3] /= h
                image=image.astype(np.uint8)
                image = cv2.resize(image, (cfg.DATA.win, cfg.DATA.hin))

                boxes_[:, 0] *= cfg.DATA.win
                boxes_[:, 1] *= cfg.DATA.hin
                boxes_[:, 2] *= cfg.DATA.win
                boxes_[:, 3] *= cfg.DATA.hin
                boxes = np.concatenate([boxes_, klass_], axis=1)
        else:
            boxes_=boxes[:, 0:4]
            klass_=boxes[:, 4:]
            image, shift_x, shift_y = Fill_img(image, target_width=cfg.DATA.win, target_height=cfg.DATA.hin)
            boxes_[:, 0:4] = boxes_[:, 0:4] + np.array([shift_x, shift_y, shift_x, shift_y], dtype='float32')
            h, w, _ = image.shape
            boxes_[:, 0] /= w
            boxes_[:, 1] /= h
            boxes_[:, 2] /= w
            boxes_[:, 3] /= h
            image = image.astype(np.uint8)
            image = cv2.resize(image, (cfg.DATA.win, cfg.DATA.hin))

            boxes_[:, 0] *= cfg.DATA.win
            boxes_[:, 1] *= cfg.DATA.hin
            boxes_[:, 2] *= cfg.DATA.win
            boxes_[:, 3] *= cfg.DATA.hin
            image = image.astype(np.uint8)
            boxes = np.concatenate([boxes_, klass_], axis=1)
        if cfg.TRAIN.vis:
            for __box in boxes:
                cv2.rectangle(image, (int(__box[0]), int(__box[1])),
                              (int(__box[2]), int(__box[3])), (255, 0, 0), 4)

        ###cove the small faces
        boxes_clean = []
        for i in range(boxes.shape[0]):
            box = boxes[i]

            if (box[3] - box[1]) < cfg.DATA.cover_small_face or (box[2] - box[0]) < cfg.DATA.cover_small_face:
                #image[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :] = 0
                continue
            else:
                boxes_clean.append(box)
        boxes=np.array(boxes_clean)

        if boxes.shape[0]==0 or np.sum(image)==0:
            boxes_ = np.array([[0, 0, 100, 100]])
            klass_ = np.array([0])
        else:
            boxes_=np.array(boxes[:,0:4],dtype=np.float32)
            klass_=np.array(boxes[:,4],dtype=np.int64)

        boxes_refine=boxes_


    except:
        logger.warn('there is an err with %s' % fname)
        traceback.print_exc()
        image = np.zeros(shape=(cfg.DATA.hin, cfg.DATA.win, 3), dtype=np.float32)
        boxes_refine = np.array([[0, 0, 100, 100]])
        klass_ = np.array([0])

    all_boxes,all_labels =produce_target(image,boxes_refine,klass_)
    return image,all_boxes,all_labels


def _map_fn(dp,is_training=True):
    fname, annos = dp
    ret=_data_aug_fn(fname,annos,is_training)
    return ret


if __name__=='__main__':
    image = np.zeros(shape=(cfg.DATA.hin, cfg.DATA.win, 3), dtype=np.float32)
    boxes_refine = np.array([[0,0,100,100]])
    klass = np.array([1])




