import cv2
import os
import time
import numpy as np
from api.face_detector import FaceDetector
from tools.to_lableimg import to_xml
from tools.to_labelme import to_json
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
detector = FaceDetector('./model/detector.pb')

import os
def GetFileList(dir, fileList):
    newDir = dir
    if os.path.isfile(dir):
        fileList.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            #如果需要忽略某些文件夹，使用以下代码
            # if s == "pts":
            #     continue
            newDir=os.path.join(dir,s)
            GetFileList(newDir, fileList)
    return fileList


def facedetect():
    count = 0
    data_dir = '/home/lz/下载'
    pics = []
    GetFileList(data_dir,pics)

    pics = [x for x in pics if 'jpg' in x or 'png' in x]
    #pics.sort()

    for pic in pics:

        img=cv2.imread(pic)

        img_show = img.copy()

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        star=time.time()
        boxes=detector(img,0.5)
        #print('one iamge cost %f s'%(time.time()-star))
        #print(boxes.shape)
        #print(boxes)
        ################toxml or json


        print(boxes.shape[0])
        if boxes.shape[0]==0:
            print(pic)
        for box_index in range(boxes.shape[0]):

            bbox = boxes[box_index]

            cv2.rectangle(img_show, (int(bbox[0]), int(bbox[1])),
                          (int(bbox[2]), int(bbox[3])), (255, 0, 0), 4)
            # cv2.putText(img_show, str(bbox[4]), (int(bbox[0]), int(bbox[1]) + 30),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1,
            #             (255, 0, 255), 2)
            #
            # cv2.putText(img_show, str(int(bbox[5])), (int(bbox[0]), int(bbox[1]) + 40),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1,
            #             (0, 0, 255), 2)


        cv2.namedWindow('res',0)
        cv2.imshow('res',img_show)
        cv2.waitKey(0)
    print(count)


if __name__=='__main__':
    facedetect()
