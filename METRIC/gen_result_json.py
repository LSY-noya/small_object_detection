import os
import json
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='divide objects by size')
parser.add_argument('-r', '--result', help='result dir')
parser.add_argument('-o', '--output', help='output dir')
args = parser.parse_args()

result_dic = {"imgs": {}}

type45 = "pn,pne,i5,p11,pl40,po,pl50,io,pl80,pl60,p26,i4,pl100,pl30,il60,i2,pl5,w57,p5,p10,pl120,il80,ip,p23,pr40,ph4.5,w59,p3,w55,pm20,p12,pg,pl70,pl20,pm55,il100,w13,p19,p27,ph4,pm30,wo,ph5,w32,p6"
type45 = type45.split(',')

files = os.listdir(args.result)
for file_i in files:
    category = file_i.split('.')[0].split('_')[1]
    path = os.path.join(args.result, file_i)
    with open(path) as f:
        data = f.readlines()
    data = list(map(lambda x: x.strip('\n').split(' '), data))
    for dataline in data:
        if float(dataline[1]) > 0.08:
            # objects {"category": "", "bbox": {"xmin": 924.0, "ymin": 1132.0, "ymax": 1177.3333, "xmax": 966.6667}}
            xmin = max(float(int(float(dataline[2]) * 2048)), 0.0)
            ymin = max(float(int(float(dataline[3]) * 2048)), 0.0)
            xmax = min(float(int(float(dataline[4]) * 2048)), 2048.0)
            ymax = min(float(int(float(dataline[5]) * 2048)), 2048.0)
            obj = {"category": category, "bbox": {
                "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}}
            if str(int(dataline[0])) in result_dic["imgs"].keys():
                result_dic["imgs"][str(int(dataline[0]))
                                   ]["objects"].append(obj)
            else:
                result_dic["imgs"][str(int(dataline[0]))] = {"objects": []}
                result_dic["imgs"][str(int(dataline[0]))
                                   ]["objects"].append(obj)

with open(os.path.join(args.output, 'new.json'), 'w') as f:
    json.dump(result_dic, f)
