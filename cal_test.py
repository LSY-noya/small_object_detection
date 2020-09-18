import os
import argparse

parser = argparse.ArgumentParser(description = 'Generate a new dataset')
parser.add_argument('-t', '--conf_thres', type=float, help = 'confidence threshold')
args = parser.parse_args()

with open('test_result.txt') as f:
    trainval = f.readlines()
trainval = list(map(lambda x:x.strip('\n').split(' '), trainval))

num = 0
for trainvalline in trainval:
    if float(trainvalline[1]) > args.conf_thres:
        num += 1

print(num)