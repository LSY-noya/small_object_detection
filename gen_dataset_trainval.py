import os
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt

type48 = "pn,pne,i5,p11,pl40,po,pl50,io,pl80,pl60,p26,i4,pl100,pl30,il60,i2,pl5,w57,p5,p10,pl120,il80,ip,p23,pr40,ph4.5,w59,p3,w55,pm20,p12,pg,pl70,pl20,pm55,il100,w13,p19,p27,ph4,pm30,wo,ph5,w32,p6,pb,wb,ib"
type48 = type48.split(',')

iou_threshold = 0.5

parser = argparse.ArgumentParser(description = 'Generate a new dataset')
parser.add_argument('-r', '--result', help = 'result dir')
parser.add_argument('-d', '--dataset', help = 'TT100k 45 dataset dir')
parser.add_argument('-a', '--annotation', help = 'Annotation dir')
parser.add_argument('-w', '--write', help = 'write dir')
args = parser.parse_args()

images_dir = os.path.join(args.dataset, 'JPEGImages')
annotation_dir = os.path.join(args.annotation, 'Annotations')

with open(os.path.join(args.result, 'tt100k_trainval_o.txt')) as f:
    trainval = f.readlines()
trainval = list(map(lambda x:x.strip('\n').split(' '), trainval))

pic_num = 'abc'

for m, trainvalline in enumerate(trainval):
    if pic_num != trainvalline[0]:
        hit = 0     # 用来图片编号
        pic = np.array(cv2.imread(os.path.join(images_dir, trainvalline[0]+'.jpg')))
        pic_num = trainvalline[0]
        with open(os.path.join(annotation_dir, trainvalline[0]+'.txt')) as f1:
            gt = f1.readlines()
        gt = list(map(lambda x:x.strip('\n').split(' '), gt))
    xmin = max(int(float(trainvalline[2])*2048), 0)
    ymin = max(int(float(trainvalline[3])*2048), 0)
    xmax = min(int(float(trainvalline[4])*2048), 2048)
    ymax = min(int(float(trainvalline[5])*2048), 2048)
    flag = 1 # flag = 1 表示图片还未切，做为是否切图标志
    bbox_d = [float(trainvalline[2]), float(trainvalline[3]), float(trainvalline[4]), float(trainvalline[5])] 
    for n, gtline in enumerate(gt):
        bbox_g = [max(float(gtline[2])/2048,0), max(float(gtline[3])/2048,0), min(float(gtline[4])/2048,1), min(float(gtline[5])/2048,1)]
        area_d = (bbox_d[2] - bbox_d[0]) * (bbox_d[3] - bbox_d[1])
        area_g = (bbox_g[2] - bbox_g[0]) * (bbox_g[3] - bbox_g[1])
        # Coincident area
        bbox_i = [max(bbox_d[0], bbox_g[0]), max(bbox_d[1], bbox_g[1]), min(bbox_d[2], bbox_g[2]), min(bbox_d[3], bbox_g[3])]
        if bbox_i[2] - bbox_i[0] > 0 and bbox_i[3] - bbox_i[1] > 0:
            area_i = (bbox_i[2] - bbox_i[0]) * (bbox_i[3] - bbox_i[1])
            iou = area_i / (area_d + area_g - area_i)
        else:
            iou = 0.

        if iou >= iou_threshold:
            flag = 0
            if gtline[1] in type48:
                pic_patch = pic[ymin:ymax, xmin:xmax]
                cv2.imwrite(os.path.join(args.write, pic_num + '_' + str(hit) + '.jpg'), pic_patch)
                with open('trainval_result.txt', 'a') as f2:
                    f2.write('pic_trainval/' + pic_num + '_' + str(hit) + '.jpg ')
                    f2.write(gtline[1] + ' ' + trainvalline[1] + '\n')
                hit = hit + 1
            else:
                # 如果iou重叠大于0.5但是不在45类中，则一定是其他的目标。48类情况此分支无效。
                pic_patch = pic[ymin:ymax, xmin:xmax]
                cv2.imwrite(os.path.join(args.write, pic_num + '_' + str(hit) + '.jpg'), pic_patch)
                with open('trainval_result.txt', 'a') as f2:
                    f2.write('pic_trainval/' + pic_num + '_' + str(hit) + '.jpg ')
                    f2.write('z' + ' ' + trainvalline[1] + '\n')
                hit = hit + 1
    if flag:
        pic_patch = pic[ymin:ymax, xmin:xmax]
        cv2.imwrite(os.path.join(args.write, pic_num + '_' + str(hit) + '.jpg'), pic_patch)
        with open('trainval_result.txt', 'a') as f2:
            f2.write('pic_trainval/' + pic_num + '_' + str(hit) + '.jpg ')
            f2.write('z' + ' ' + trainvalline[1] + '\n')
        hit = hit + 1
