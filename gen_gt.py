import os
import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser(description = 'Generate a new dataset')
parser.add_argument('-d', '--data', help = 'data source dir')
parser.add_argument('-w', '--write', help = 'write dir')
args = parser.parse_args()

type45 = "pn,pne,i5,p11,pl40,po,pl50,io,pl80,pl60,p26,i4,pl100,pl30,il60,i2,pl5,w57,p5,p10,pl120,il80,ip,p23,pr40,ph4.5,w59,p3,w55,pm20,p12,pg,pl70,pl20,pm55,il100,w13,p19,p27,ph4,pm30,wo,ph5,w32,p6"
type45 = type45.split(',')

trainvalpic_dir = os.path.join(args.write, 'trainval_gt')
with open(os.path.join(args.data, 'trainval.txt')) as f:
    trainval = f.readlines()
trainval = list(map(lambda x:x.strip('\n').split(' '), trainval))

for m, trainvalline in enumerate(trainval):
    pic = np.array(cv2.imread(os.path.join(args.data, trainvalline[0])))
    with open(os.path.join(args.data, trainvalline[1])) as f1:
        ann = f1.readlines()
    ann = list(map(lambda x:x.strip('\n').split(' '), ann))
    for n, annline in enumerate(ann):
        label_num = type45.index(annline[1])
        xmin = max(int(float(annline[2])), 0)
        ymin = max(int(float(annline[3])), 0)
        xmax = min(int(float(annline[4])), 2048)
        ymax = min(int(float(annline[5])), 2048)
        if label_num >= 13:
            pic_patch = pic[ymin:ymax, xmin:xmax]
            cv2.imwrite(os.path.join(trainvalpic_dir, annline[0] + '_' + str(n) + '.jpg'), pic_patch)
            with open(os.path.join(args.write, 'trainval_gt.txt'), 'a') as f2:
                f2.write('trainval_gt/' + annline[0] + '_' + str(n) + '.jpg ')
                f2.write(annline[1] + '\n')
