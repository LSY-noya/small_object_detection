# 用途：将trainval_gt较少的目标与trainval_result融合，并对trainval_result多的目标欠采样，以缓解不均衡

import os
import random

type45 = "pn,pne,i5,p11,pl40,po,pl50,io,pl80,pl60,p26,i4,pl100,pl30,il60,i2,pl5,w57,p5,p10,pl120,il80,ip,p23,pr40,ph4.5,w59,p3,w55,pm20,p12,pg,pl70,pl20,pm55,il100,w13,p19,p27,ph4,pm30,wo,ph5,w32,p6,pb,wb,ib,z"
type45 = type45.split(',')

# result    sub2    sub3
with open('trainval_result_0p08.txt') as f:
    result = f.readlines()
result = list(map(lambda x:x.strip('\n').split(' '), result))

# gt        sub2    sub3
with open('../../trainval_gt.txt') as f:
    gt = f.readlines()
gt = list(map(lambda x:x.strip('\n').split(' '), gt))

# extra             sub3
with open('../../trainval_extra.txt') as f:
    extra = f.readlines()
extra = list(map(lambda x:x.strip('\n').split(' '), extra))


# write sub2
with open('trainval_result_0p08_2.txt','a') as f:
    for resultline in result:
        f.write(resultline[0] + ' ' + resultline[1] + '\n')

with open('trainval_result_0p08_2.txt','a') as f:
    for resultline in gt:
        f.write('../../' + resultline[0] + ' ' + resultline[1] + '\n')


# write sub3
with open('trainval_result_0p08_3.txt','a') as f:
    for resultline in result:
        f.write(resultline[0] + ' ' + resultline[1] + '\n')

with open('trainval_result_0p08_3.txt','a') as f:
    for resultline in gt:
        f.write('../../' + resultline[0] + ' ' + resultline[1] + '\n')

with open('trainval_result_0p08_3.txt','a') as f:
    for resultline in extra:
        f.write('../../' + resultline[0] + ' ' + resultline[1] + '\n')