import os
import argparse

type48 = "pn,pne,i5,p11,pl40,po,pl50,io,pl80,pl60,p26,i4,pl100,pl30,il60,i2,pl5,w57,p5,p10,pl120,il80,ip,p23,pr40,ph4.5,w59,p3,w55,pm20,p12,pg,pl70,pl20,pm55,il100,w13,p19,p27,ph4,pm30,wo,ph5,w32,p6,pb,wb,ib,z"
type48 = type48.split(',')

class_num = [0] * len(type48)

parser = argparse.ArgumentParser(description = 'Generate a new dataset')
parser.add_argument('-t', '--txt', help = 'txt path')
args = parser.parse_args()

file_path = args.txt

with open(file_path) as f:
    trainval = f.readlines()
trainval = list(map(lambda x:x.strip('\n').split(' '), trainval))

for trainvalline in trainval:
    idx = type48.index(trainvalline[1])
    class_num[idx] += 1

num_obj = 0
for m in range(49):
    if type48[m] != 'z':
        num_obj += class_num[m]
print(class_num[0:45])
print('num_obj:\t' + str(num_obj))
print('background:\t' + str(class_num[-1]))
