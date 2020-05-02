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

type45 = "pn,pne,i5,p11,pl40,po,pl50,io,pl80,pl60,p26,i4,pl100,pl30,il60,i2,pl5,w57,p5,p10,pl120,il80,ip,p23,pr40,ph4.5,w59,p3,w55,pm20,p12,pg,pl70,pl20,pm55,il100,w13,p19,p27,ph4,pm30,wo,ph5,w32,p6"
type45 = type45.split(',')

type45 = ['i2', 'i4', 'i5', 'il100', 'il60', 'il80', 'io', 'ip', 'p10', 'p11', 'p12', 'p19', 'p23', 'p26', 'p27', 'p3', 'p5', 'p6', 'pg', 'ph4', 'ph4.5', 'ph5', 'pl100', 'pl120', 'pl20', 'pl30', 'pl40', 'pl5', 'pl50', 'pl60', 'pl70', 'pl80', 'pm20', 'pm30', 'pm55', 'pn', 'pne', 'po', 'pr40', 'w13', 'w32', 'w55', 'w57', 'w59', 'wo']

# type45 = ['small', 'medium', 'large']
# type_thres = [0.097, 0.1, 0.115, 0.108, 0.0755, 0.115, 0.09894, 0.09, 0.13, 0.057, 0.095, 0.14, 0.03, 0.07, 0.125, 0.1, 0.06, 0.104, 0.08, 0.04, 0.1, 0.3, 0.1, 0.12, 0.3, 0.04, 0.1, 0.1, 0.1, 0.04, 0.08, 0.051, 0.1, 0.1, 0.06, 0.25, 0.08, 0.2, 0.1, 0.1, 0.6, 0.27, 0.12, 0.4, 0.1]

type_thres = [0.1, 0.04, 0.03, 0.035, 0.035, 0.03, 0.03, 0.03, 0.158, 0.03, 0.03, 0.03, 0.1, 0.05, 0.2, 0.07, 0.037, 0.053, 0.03, 0.06, 0.4, 0.07, 0.03, 0.03, 0.1, 0.5, 0.6, 0.5, 0.2, 0.03, 0.2, 0.045, 0.2, 0.2, 0.07, 0.3, 0.1, 0.2, 0.03, 0.03, 0.1, 0.2, 0.03, 0.03, 0.2]
type_thres = [0.05] * 45
# type_thres = [0.032, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.045, 0.01, 0.01, 0.0125, 0.01, 0.035, 0.04, 0.05, 0.01, 0.1, 0.3, 0.01, 0.05, 0.3, 0.1, 0.4, 0.1, 0.01, 0.01, 0.08, 0.04, 0.04, 0.06, 0.06, 0.01, 0.08, 0.2, 0.01,     0.01, 0.2, 0.27, 0.1, 0.1, 0.01]
# type45 = ['small', 'medium', 'large']

iou_threshold = 0.5

# 检测和真值都是以tt100k开头的
mAP = []
mP = []
for category in type45:
    # 对检测进行加载和排序
    detection_file = open(os.path.join(args.detection, 'tt100k_%s.txt' % category))
    detection_result = detection_file.readlines()
    detection_result = list(map(lambda x:x.strip('\n').split(' '), detection_result))
    detection_file.close()
    # detection_result的格式是id score xmin ymin xmax yman
    detection_result = sorted(detection_result, key = lambda x: float(x[1]), reverse = True)
    point = 0
    for m in range(len(detection_result)-1, -1, -1):
        if float(detection_result[m][1]) > type_thres[type45.index(category)]:
            point = m + 1
            break
    detection_result = detection_result[:point]
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
    for i, detection in enumerate(detection_result):
        # 计算是否存在满足阈值以上的检测框
        iou_max = 0.
        index_max = -1
        for j, groundtruth in enumerate(groundtruth_result):
            # 判断两者是否是同一张图
            if detection[0] != groundtruth[0]:
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
    print('R for %s: %f' % (category, recall_curve[-1]))
    print('P for %s: %f' % (category, precision_curve[-1]))
    mP.append(precision_curve[-1])
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
print('detection_eval_mP = %f' % (sum(mP)/len(mP)))
