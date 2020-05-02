import os
import copy
import argparse

parser = argparse.ArgumentParser(description = 'divide objects by size')
parser.add_argument('-r', '--result', help = 'result dir')
parser.add_argument('-o', '--output', help = 'output dir')
parser.add_argument('-m', '--mode', type = int, help = 'output dir')
args = parser.parse_args()

# mode=0 分gt   mode=1 分result
mode =args.mode

small = []
medium = []
large = []

files = os.listdir(args.result)
for file_i in files:
    path = os.path.join(args.result, file_i)
    with open(path) as f:
        data = f.readlines()
    data = list(map(lambda x:x.strip('\n').split(' '), data))
    for dataline in data:
        boxsize = max((float(dataline[3 + mode]) - float(dataline[1 + mode])) * 2048, (float(dataline[4 + mode]) - float(dataline[2 + mode])) * 2048)
        boxsize = int(boxsize)
        if boxsize <= 32:
            small.append(copy.deepcopy(dataline))
        elif boxsize > 32 and boxsize <= 96:
            medium.append(copy.deepcopy(dataline))
        elif boxsize > 96 and boxsize <= 400:
            large.append(copy.deepcopy(dataline))

if mode:
    # 写result
    with open(os.path.join(args.output, 'tt100k_small.txt'), 'a') as f:
        for m in small:
            f.write(m[0] + ' ')
            f.write(m[1] + ' ')
            f.write(m[2] + ' ')
            f.write(m[3] + ' ')
            f.write(m[4] + ' ')
            f.write(m[5] + '\n')
    with open(os.path.join(args.output, 'tt100k_medium.txt'), 'a') as f:
        for m in medium:
            f.write(m[0] + ' ')
            f.write(m[1] + ' ')
            f.write(m[2] + ' ')
            f.write(m[3] + ' ')
            f.write(m[4] + ' ')
            f.write(m[5] + '\n')
    with open(os.path.join(args.output, 'tt100k_large.txt'), 'a') as f:
        for m in large:
            f.write(m[0] + ' ')
            f.write(m[1] + ' ')
            f.write(m[2] + ' ')
            f.write(m[3] + ' ')
            f.write(m[4] + ' ')
            f.write(m[5] + '\n')
else:
    with open(os.path.join(args.output, 'tt100k_small.txt'), 'a') as f:
        for m in small:
            f.write(m[0] + ' ')
            f.write(m[1] + ' ')
            f.write(m[2] + ' ')
            f.write(m[3] + ' ')
            f.write(m[4] + '\n')
    with open(os.path.join(args.output, 'tt100k_medium.txt'), 'a') as f:
        for m in medium:
            f.write(m[0] + ' ')
            f.write(m[1] + ' ')
            f.write(m[2] + ' ')
            f.write(m[3] + ' ')
            f.write(m[4] + '\n')
    with open(os.path.join(args.output, 'tt100k_large.txt'), 'a') as f:
        for m in large:
            f.write(m[0] + ' ')
            f.write(m[1] + ' ')
            f.write(m[2] + ' ')
            f.write(m[3] + ' ')
            f.write(m[4] + '\n')
