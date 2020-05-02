#-*-coding:utf-8-*-

import os
import argparse
parser = argparse.ArgumentParser(description = 'Generate groundtruth')
parser.add_argument('-t', '--test', help = 'test.txt dir')
args = parser.parse_args()

# annotation:id class xmin ymin xmax ymax
with open(os.path.join(args.test, 'test.txt')) as f:
    gt_info = f.readlines()
gt_info = list(map(lambda x:x.strip('\n').split(' '), gt_info))

for m, annotation in enumerate(gt_info):
    ann_path = os.path.join(args.test, annotation[1])
    with open(ann_path) as f1:
        ann_temp = f1.readlines()
    ann_temp = list(map(lambda x:x.strip('\n').split(' '), ann_temp))
    for n, ann_line in enumerate(ann_temp):
        with open('tt100k_%s_3cate.txt' % ann_line[1][0],'a') as f2:
            f2.write(ann_line[0]+' ')
            f2.write('{:.10f}'.format(float(ann_line[2])/2048) + ' ')
            f2.write('{:.10f}'.format(float(ann_line[3])/2048) + ' ')
            f2.write('{:.10f}'.format(float(ann_line[4])/2048) + ' ')
            f2.write('{:.10f}'.format(float(ann_line[5])/2048) + '\n')
