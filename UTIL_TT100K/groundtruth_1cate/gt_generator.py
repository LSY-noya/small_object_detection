#-*-coding:utf-8-*-

import os
import argparse
parser = argparse.ArgumentParser(description = 'Generate groundtruth')
parser.add_argument('-t', '--test', help = 'test.txt dir')
args = parser.parse_args()

type42 = "pn,pne,i5,p11,pl40,pl50,pl80,pl60,p26,i4,pl100,pl30,il60,i2,pl5,w57,p5,p10,pl120,il80,ip,p23,pr40,ph4.5,w59,p3,w55,pm20,p12,pg,pl70,pl20,pm55,il100,w13,p19,p27,ph4,pm30,ph5,w32,p6"
type42 = type42.split(',')

type45 = "pn,pne,i5,p11,pl40,po,pl50,io,pl80,pl60,p26,i4,pl100,pl30,il60,i2,pl5,w57,p5,p10,pl120,il80,ip,p23,pr40,ph4.5,w59,p3,w55,pm20,p12,pg,pl70,pl20,pm55,il100,w13,p19,p27,ph4,pm30,wo,ph5,w32,p6"
type45 = type45.split(',')

# annotation:id class xmin ymin xmax ymax
with open(os.path.join(args.test, 'test.txt')) as f:
    gt_info = f.readlines()
gt_info = list(map(lambda x:x.strip('\n').split(' '), gt_info))

for m, annotation in enumerate(gt_info):
    ann_path = os.path.join(args.test, annotation[1])
    with open(ann_path) as f1:
        ann_temp = f1.readlines()
    ann_temp = list(map(lambda x:x.strip('\n').split(' '), ann_temp))
    with open('tt100k_o.txt','a') as f2:
        for n, ann_line in enumerate(ann_temp):
#             if ann_line[1] in type45:
            f2.write(ann_line[0]+' ')
            f2.write('{:.10f}'.format(float(ann_line[2])/2048) + ' ')
            f2.write('{:.10f}'.format(float(ann_line[3])/2048) + ' ')
            f2.write('{:.10f}'.format(float(ann_line[4])/2048) + ' ')
            f2.write('{:.10f}'.format(float(ann_line[5])/2048) + '\n')
