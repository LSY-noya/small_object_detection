import os
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt



parser = argparse.ArgumentParser(description = 'Generate test dataset')
parser.add_argument('-r', '--result', help = 'result dir')
parser.add_argument('-d', '--dataset', help = 'TT100k 45 dataset dir')
parser.add_argument('-w', '--write', help = 'write dir')
args = parser.parse_args()

images_dir = os.path.join(args.dataset, 'JPEGImages')
with open(os.path.join(args.result, 'tt100k_o.txt')) as f:
    test = f.readlines()
test = list(map(lambda x:x.strip('\n').split(' '), test))

pic_num = 'abc' # initialization
for m, testline in enumerate(test):
    if testline[0] != pic_num:
        count = 0
        pic = np.array(cv2.imread(os.path.join(images_dir, testline[0]+'.jpg')))
        pic_num = testline[0]
    if float(testline[1]) > 0.01:
        xmin = max(int(float(testline[2])*2048), 0)
        ymin = max(int(float(testline[3])*2048), 0)
        xmax = min(int(float(testline[4])*2048), 2048)
        ymax = min(int(float(testline[5])*2048), 2048)
        pic_patch = pic[ymin:ymax, xmin:xmax]
        cv2.imwrite(os.path.join(args.write, pic_num + '_' + str(count) + '.jpg'), pic_patch)
        with open('test_result.txt','a') as f:
            f.write('pic_test/' + pic_num + '_' + str(count) + '.jpg ')
            f.write(testline[1] + '\n')
    count = count + 1
