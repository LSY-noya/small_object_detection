#-*-coding:utf-8-*-
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description = 'compute mAP')
parser.add_argument('-d', '--detection', help = 'detection output dir')
parser.add_argument('-g', '--groundtruth', help = 'groundtruth dir')
args = parser.parse_args()

type1 = ['o']
iou_threshold = 0.5

thres = list(np.arange(0, 1.01, 0.01))

# 对检测进行加载和排序
detection_file = open(os.path.join(args.detection, 'tt100k_o.txt'))
detection_result = detection_file.readlines()
detection_result = list(map(lambda x:x.strip('\n').split(' '), detection_result))
detection_file.close()
# detection_result的格式是id score xmin ymin xmax yman
detection_result = sorted(detection_result, key = lambda x: float(x[1]), reverse = True)
# 对groundtruth进行加载
groundtruth_file = open(os.path.join(args.groundtruth, 'tt100k_o.txt'))
groundtruth_result = groundtruth_file.readlines()
groundtruth_result = np.array(list(map(lambda x:x.strip('\n').split(' '), groundtruth_result)))
groundtruth_file.close()
# groundtruth_result的格式是id xmin ymin xmax ymax
total_positive = len(groundtruth_result)

# 计算recall--threshold曲线
recall = []
for threshold in thres:
    predict_positive = [0] * total_positive
    for m, detection in enumerate(detection_result):
        if threshold < float(detection[1]):
            gt_pos = np.where(detection[0].split('_')[0] == groundtruth_result)[0]
            for n in gt_pos:
                groundtruth = groundtruth_result[n]
                bbox_d = [float(detection[2]), float(detection[3]), float(detection[4]), float(detection[5])]
                bbox_g = [float(groundtruth[1]), float(groundtruth[2]), float(groundtruth[3]), float(groundtruth[4])]
                area_d = (bbox_d[2] - bbox_d[0]) * (bbox_d[3] - bbox_d[1])
                area_g = (bbox_g[2] - bbox_g[0]) * (bbox_g[3] - bbox_g[1])
                # 计算重合面积
                bbox_i = [max(bbox_d[0], bbox_g[0]), max(bbox_d[1], bbox_g[1]), min(bbox_d[2], bbox_g[2]), min(bbox_d[3], bbox_g[3])]
                if bbox_i[2] - bbox_i[0] > 0 and bbox_i[3] - bbox_i[1] > 0:
                    area_i = (bbox_i[2] - bbox_i[0]) * (bbox_i[3] - bbox_i[1])
                    iou = area_i / (area_d + area_g - area_i)
                else:
                    iou = 0.
                if iou > iou_threshold:
                    predict_positive[n] = 1
    tp = sum(predict_positive)
    recall.append(tp / total_positive)
    print(str(threshold) + '_recall:' + str(tp / total_positive))
    with open('recall_SSD.txt','a') as f:
        f.write('{:.10f}'.format(tp / total_positive) + '\n')

plt.plot(thres, recall)
plt.savefig('./recall_SSD.jpg')
