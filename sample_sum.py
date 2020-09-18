import os
import argparse

type48 = "pn,pne,i5,p11,pl40,po,pl50,io,pl80,pl60,p26,i4,pl100,pl30,il60,i2,pl5,w57,p5,p10,pl120,il80,ip,p23,pr40,ph4.5,w59,p3,w55,pm20,p12,pg,pl70,pl20,pm55,il100,w13,p19,p27,ph4,pm30,wo,ph5,w32,p6,pb,wb,ib,z"
type48 = type48.split(',')

class_num = [0] * len(type48)

parser = argparse.ArgumentParser(description = 'Generate a new dataset')
parser.add_argument('-t', '--conf_thres', help = 'confidence threshold')
parser.add_argument('-m', '--mode', help = '0:not write, 1:write', type=int, default=0)
args = parser.parse_args()

conf_thres = float(args.conf_thres)
write_name = args.conf_thres.split('.')
write_name = write_name[0] + 'p' + write_name[1]

with open('trainval_result.txt') as f:
    trainval = f.readlines()
trainval = list(map(lambda x:x.strip('\n').split(' '), trainval))

for trainvalline in trainval:
    if float(trainvalline[2]) > conf_thres:
        idx = type48.index(trainvalline[1])
        class_num[idx] += 1
        if args.mode:
            with open('trainval_result_%s.txt' % write_name, 'a') as f:
                f.write(trainvalline[0] + ' ')
                f.write(trainvalline[1] + ' ')
                f.write(trainvalline[2] + '\n')

num_obj = 0
for m in range(len(type48)):
    # print(type48[m] + ':' + str(class_num[m]) + '\n')
    if type48[m] != 'z':
        num_obj += class_num[m]
print(class_num)
print(max(class_num[0:45]))
print(min(class_num[0:45]))
print('num_obj:\t' + str(num_obj))
print('background:\t' + str(class_num[-1]))
