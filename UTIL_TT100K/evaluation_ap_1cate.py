#!/home/sunhanbo/software/anaconda3/bin/python
#-*-coding:utf-8-*-
# 本文件是用来对caffe模型的输出结果进行mAP计算的脚本
# 因为之后有可能会更改为双段式检测，因此需要本脚本来结果进行分析
import os
import argparse
parser = argparse.ArgumentParser(description = 'compute mAP')
parser.add_argument('-d', '--detection', help = 'detection output dir')
parser.add_argument('-g', '--groundtruth', help = 'groundtruth dir')
parser.add_argument('-m', '--method', choices = ['11point', 'MaxIntegral', 'Integral'], help = 'AP method')
args = parser.parse_args()

type1 = ['o']

iou_threshold = 0.5

# 检测和真值都是以tt100k开头的
mAP = []
for category in type1:
    # 对检测进行加载和排序
    detection_file = open(os.path.join(args.detection, 'tt100k_%s.txt' % category))
    detection_result = detection_file.readlines()
    detection_result = list(map(lambda x:x.strip('\n').split(' '), detection_result))
    detection_file.close()
    # detection_result的格式是id score xmin ymin xmax yman
    detection_result = sorted(detection_result, key = lambda x: float(x[1]), reverse = True)
    # 对groundtruth进行加载
    groundtruth_file = open(os.path.join(args.groundtruth, 'tt100k_%s.txt' % category))
    groundtruth_result = groundtruth_file.readlines()
    groundtruth_result = list(map(lambda x:x.strip('\n').split(' '), groundtruth_result))
    groundtruth_file.close()
    # groundtruth_result的格式是id xmin ymin xmax ymax
    total_predict = len(detection_result)
    total_positive = len(groundtruth_result)
    assert total_predict > 0
    assert total_positive > 0
    predict_positive = [0] * total_positive
    # 计算准确率和回归曲线
    precision_curve = []
    recall_curve = []
    num_front = 0
    for i, detection in enumerate(detection_result):
        # 计算是否存在满足阈值以上的检测框
        iou_max = 0.
        index_max = -1
        for j, groundtruth in enumerate(groundtruth_result):
            # 判断两者是否是同一张图
            if detection[0].split('_')[0] != groundtruth[0]:
                continue
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
            if iou >= iou_threshold:
                if iou > iou_max:
                    iou_max = iou
                    index_max = j
        # 对框进行判断和处理
        if index_max != -1:
            predict_positive[index_max] = 1
        tp = sum(predict_positive)
        precision_curve.append(tp / float(i + 1))
        recall_curve.append(tp / total_positive)
    # 通过上述两个曲线对mAP进行计算

    print(len(predict_positive))
    print(sum(predict_positive))
    
    if args.method == '11point':
        # 11point按照recall来划分
        max_precs = [0.] * 11
        start_idx = total_predict - 1
        for j in range(10, -1, -1):
            for i in range(start_idx, -1, -1):
                if recall_curve[i] < (j / 10.):
                    start_idx = i
                    if j > 0:
                        max_precs[j - 1] = max_precs[j]
                    break
                else:
                    if max_precs[j] < precision_curve[i]:
                        max_precs[j] = precision_curve[i]
        print('AP for %s: %f' % (category, sum(max_precs) / 11.))
        mAP.append(sum(max_precs) / 11.)
    elif args.method == 'MaxIntegral':
        ap = 0.
        cur_recall = recall_curve[-1]
        cur_precision = precision_curve[-1]
        for i in range(total_predict - 2, -1, -1):
            cur_precision = max(cur_precision, precision_curve[i])
            ap += cur_precision * (cur_recall - recall_curve[i])
            cur_recall = recall_curve[i]
        ap += cur_recall * cur_precision
        print('AP for %s: %f' % (category, ap))
        mAP.append(ap)
    elif args.method == 'Integral':
        ap = 0.
        pre_recall = 0.
        for i in range(total_predict):
            ap += precision_curve[i] * (recall_curve[i] - pre_recall)
            pre_recall = recall_curve[i]
        print('AP for %s: %f' % (category, ap))
        mAP.append(ap)
    else:
        raise NotImplementedError
print('detection_eval = %f' % (sum(mAP)/len(mAP)))
