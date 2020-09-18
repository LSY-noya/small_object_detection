import os
import argparse
import numpy as np

parser = argparse.ArgumentParser(description = 'gen lenet5 result')
parser.add_argument('-o', '--output', help = 'output dir')
args = parser.parse_args()

type45 = "pn,pne,i5,p11,pl40,po,pl50,io,pl80,pl60,p26,i4,pl100,pl30,il60,i2,pl5,w57,p5,p10,pl120,il80,ip,p23,pr40,ph4.5,w59,p3,w55,pm20,p12,pg,pl70,pl20,pm55,il100,w13,p19,p27,ph4,pm30,wo,ph5,w32,p6"
type45 = type45.split(',')

with open('../../test_output/tt100k_o.txt') as f:
    result_1 = f.readlines()
result_1 = list(map(lambda x:x.strip('\n').split(' '), result_1))

with open('./results/test_pred.txt') as f:
    result_2 = f.readlines()
result_2 = list(map(lambda x:x.strip('\n').split(' '), result_2))

if not os.path.exists(args.output):
    os.makedirs(args.output)

for m in range(np.shape(result_1)[0]):
    if result_2[m][0] in type45:
        with open(os.path.join(args.output, 'tt100k_%s.txt' % result_2[m][0]), 'a') as f:
            f.write(result_1[m][0].split('_')[0] + ' ')
            f.write(result_1[m][1] + ' ')
            f.write(result_1[m][2] + ' ')
            f.write(result_1[m][3] + ' ')
            f.write(result_1[m][4] + ' ')
            f.write(result_1[m][5] + '\n')

if os.path.exists('./results/test_pred.txt'):
    os.remove('./results/test_pred.txt')
