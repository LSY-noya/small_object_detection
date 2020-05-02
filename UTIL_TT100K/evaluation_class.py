import os
import argparse
parser = argparse.ArgumentParser(description = 'evaluate clf mistake')
parser.add_argument('-d', '--detection', help = 'detection output dir')
parser.add_argument('-g', '--groundtruth', help = 'groundtruth dir')
parser.add_argument('-c', '--classname', help = 'class name')
args = parser.parse_args()

type45 = "pn,pne,i5,p11,pl40,po,pl50,io,pl80,pl60,p26,i4,pl100,pl30,il60,i2,pl5,w57,p5,p10,pl120,il80,ip,p23,pr40,ph4.5,w59,p3,w55,pm20,p12,pg,pl70,pl20,pm55,il100,w13,p19,p27,ph4,pm30,wo,ph5,w32,p6"
type45 = type45.split(',')

iou_threshold = 0.5

class_name = args.classname
# 对检测进行加载和排序
detection_file = open(os.path.join(args.detection, 'tt100k_%s.txt' % class_name))
detection_result = detection_file.readlines()
detection_result = list(map(lambda x:x.strip('\n').split(' '), detection_result))
detection_file.close()
# detection_result的格式是id score xmin ymin xmax yman
detection_result = sorted(detection_result, key = lambda x: float(x[1]), reverse = True)

result_c = [0] * 45

for category in type45:
    # 对groundtruth进行加载
    groundtruth_file = open(os.path.join(args.groundtruth, 'tt100k_%s.txt' % category))
    groundtruth_result = groundtruth_file.readlines()
    groundtruth_result = list(map(lambda x:x.strip('\n').split(' '), groundtruth_result))
    groundtruth_file.close()
    # groundtruth_result的格式是id xmin ymin xmax ymax
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
                result_c[type45.index(category)] += 1
                # if iou > iou_max:
                #     iou_max = iou
                #     index_max = j
        # 对框进行判断和处理
        # if index_max != -1:
        #     result_c[type45.index(category)] += 1

print(class_name + ' total num : ' + str(len(detection_result)))
for category in type45:
    print(category + ' : ' + str(result_c[type45.index(category)]))