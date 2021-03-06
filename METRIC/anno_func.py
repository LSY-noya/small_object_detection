import json
import pylab as pl
import random
import numpy as np
import cv2
import copy

type45="i2,i4,i5,il100,il60,il80,io,ip,p10,p11,p12,p19,p23,p26,p27,p3,p5,p6,pg,ph4,ph4.5,ph5,pl100,pl120,pl20,pl30,pl40,pl5,pl50,pl60,pl70,pl80,pm20,pm30,pm55,pn,pne,po,pr40,w13,w32,w55,w57,w59,wo"
type45 = type45.split(',')

def load_img(annos, datadir, imgid):
    img = annos["imgs"][imgid]
    imgpath = datadir+'/'+img['path']
    imgdata = pl.imread(imgpath)
    #imgdata = (imgdata.astype(np.float32)-imgdata.min()) / (imgdata.max() - imgdata.min())
    if imgdata.max() > 2:
        imgdata = imgdata/255.
    return imgdata

def load_mask(annos, datadir, imgid, imgdata):
    img = annos["imgs"][imgid]
    mask = np.zeros(imgdata.shape[:-1])
    mask_poly = np.zeros(imgdata.shape[:-1])
    mask_ellipse = np.zeros(imgdata.shape[:-1])
    for obj in img['objects']:
        box = obj['bbox']
        cv2.rectangle(mask, (int(box['xmin']), int(box['ymin'])), (int(box['xmax']), int(box['ymax'])), 1, -1)
        if obj.has_key('polygon') and len(obj['polygon'])>0:
            pts = np.array(obj['polygon'])
            cv2.fillPoly(mask_poly, [pts.astype(np.int32)], 1)
            # print pts
        else:
            cv2.rectangle(mask_poly, (int(box['xmin']), int(box['ymin'])), (int(box['xmax']), int(box['ymax'])), 1, -1)
        if obj.has_key('ellipse'):
            rbox = obj['ellipse']
            rbox = ((rbox[0][0], rbox[0][1]), (rbox[1][0], rbox[1][1]), rbox[2])
            print(rbox)
            cv2.ellipse(mask_ellipse, rbox, 1, -1)
        else:
            cv2.rectangle(mask_ellipse, (int(box['xmin']), int(box['ymin'])), (int(box['xmax']), int(box['ymax'])), 1, -1)
    mask = np.multiply(np.multiply(mask,mask_poly),mask_ellipse)
    return mask
    
def draw_all(annos, datadir, imgid, imgdata, color=(0,1,0), have_mask=True, have_label=True):
    img = annos["imgs"][imgid]
    if have_mask:
        mask = load_mask(annos, datadir, imgid, imgdata)
        imgdata = imgdata.copy()
        imgdata[:,:,0] = np.clip(imgdata[:,:,0] + mask*0.7, 0, 1)
    for obj in img['objects']:
        box = obj['bbox']
        cv2.rectangle(imgdata, (int(box['xmin']), int(box['ymin'])), (int(box['xmax']), int(box['ymax'])), color, 3)
        ss = obj['category']
        if obj.has_key('correct_catelog'):
            ss = ss+'->'+obj['correct_catelog']
        if have_label:
            cv2.putText(imgdata, ss, (int(box['xmin']),int(box['ymin']-10)), 0, 1, color, 2)
    return imgdata

def rect_cross(rect1, rect2):
    rect = [max(rect1[0], rect2[0]),
            max(rect1[1], rect2[1]),
            min(rect1[2], rect2[2]),
            min(rect1[3], rect2[3])]
    rect[2] = max(rect[2], rect[0])
    rect[3] = max(rect[3], rect[1])
    return rect

def rect_area(rect):
    return float(max(0.0, (rect[2]-rect[0])*(rect[3]-rect[1])))

def calc_cover(rect1, rect2):
    crect = rect_cross(rect1, rect2)
    return rect_area(crect) / rect_area(rect2)

def calc_iou(rect1, rect2):
    crect = rect_cross(rect1, rect2)
    ac = rect_area(crect)
    a1 = rect_area(rect1)
    a2 = rect_area(rect2)
    return ac / (a1+a2-ac)

def get_refine_rects(annos, raw_rects, minscore=20):
    cover_th = 0.5
    refine_rects = {}

    for imgid in raw_rects.keys():
        v = raw_rects[imgid]
        tv = copy.deepcopy(sorted(v, key=lambda x:-x[2]))
        nv = []
        for obj in tv:
            rect = obj[1]
            rect[2]+=rect[0]
            rect[3]+=rect[1]
            if rect_area(rect) == 0: continue
            if obj[2] < minscore: continue
            cover_area = 0
            for obj2 in nv:
                cover_area += calc_cover(obj2[1], rect)
            if cover_area < cover_th:
                nv.append(obj)
        refine_rects[imgid] = nv
    results = {}
    for imgid, v in refine_rects.items():
        objs = []
        for obj in v:
            mobj = {"bbox":dict(zip(["xmin","ymin","xmax","ymax"], obj[1])), 
                    "category":annos['types'][int(obj[0]-1)], "score":obj[2]}
            objs.append(mobj)
        results[imgid] = {"objects":objs}
    results_annos = {"imgs":results}
    return results_annos

def box_long_size(box):
    # return max(box['xmax']-box['xmin'], box['ymax']-box['ymin'])
    return (box['xmax']-box['xmin']) * (box['ymax']-box['ymin'])

def box_wide_size(box):
    return box['xmax']-box['xmin']

def eval_annos(annos_gd, annos_rt, iou=0.5, imgids=None, check_type=True, types=None, minscore=40, minboxsize=0, maxboxsize=400, match_same=True):
    ac_n, ac_c = 0,0
    rc_n, rc_c = 0,0
    if imgids==None:
        imgids = annos_rt['imgs'].keys()
    if types!=None:
        types = { t:0 for t in types } # 将types变成字典，记录每一个类的个数，e.g. {'a': 0, 'b': 0, 'c': 0, 'd': 0, 'e': 0}
    miss = {"imgs":{}}
    wrong = {"imgs":{}}
    right = {"imgs":{}}
    gt_num = 0
    for imgid in imgids:    # 选取一张图片
        # 自己加的 **********
        if imgid not in annos_rt['imgs'].keys():
            annos_rt['imgs'][imgid] = {'objects':[]}

        v = annos_rt['imgs'][imgid]
        vg = annos_gd['imgs'][imgid]
        convert = lambda objs: [ [ obj['bbox'][key] for key in ['xmin','ymin','xmax','ymax']] for obj in objs] # 把某一幅图片中的xmin， ymin， xmax， ymax提取出来。
        objs_g = vg["objects"]
        objs_r = v["objects"]
        bg = convert(objs_g)    # 图片上所有目标的gt位置
        br = convert(objs_r)    # 图片上所有目标的detection位置
        
        match_g = [-1]*len(bg)
        match_r = [-1]*len(br)
        if types!=None:
            for i in range(len(match_g)):
                if objs_g[i]['category'] not in types.keys():
                    match_g[i] = -2                     # -2代表不用计算AP。
            for i in range(len(match_r)):
                if objs_r[i]['category'] not in types.keys():
                    match_r[i] = -2
        for i in range(len(match_r)):   # 我们没有设置这一项。自动判定不用管
            if ('score' in objs_r[i].keys()) and objs_r[i]['score']<minscore:
                match_r[i] = -2
        matches = []
        for i,boxg in enumerate(bg):
            for j,boxr in enumerate(br):
                if match_g[i] == -2 or match_r[j] == -2: # 只有当两个的type都存在时才进行之后的操作。
                    continue
                if match_same and objs_g[i]['category'] != objs_r[j]['category']: continue  # 只有在类别完全一致时才会继续
                tiou = calc_iou(boxg, boxr)
                if tiou>iou:
                    matches.append((tiou, i, j))
        matches = sorted(matches, key=lambda x:-x[0])   # 对matches按照iou的从大到小排序。
        for tiou, i, j in matches:
            if match_g[i] == -1 and match_r[j] == -1:
                match_g[i] = j                          # 给出匹配位置。
                match_r[j] = i
                
        for i in range(len(match_g)):
            boxsize = box_wide_size(objs_g[i]['bbox'])              # 尺寸要求指的是较长的一边。
            erase = False
            if not (boxsize>minboxsize and boxsize<=maxboxsize):    # 不满足尺寸要求的一律不参与算AP，总的来说就是完全按照尺寸分的。左包含[min, max)
                erase = True
            #if types!=None and not types.has_key(objs_g[i]['category']):
            #    erase = True
            if erase:
                if match_g[i] >= 0:
                    match_r[match_g[i]] = -2
                match_g[i] = -2
        
        for i in range(len(match_r)):
            boxsize = box_wide_size(objs_r[i]['bbox'])
            if match_r[i] != -1: continue                           # 处理detection等于-1的情况
            if not (boxsize>minboxsize and boxsize<=maxboxsize):
                match_r[i] = -2                                     # 如果detection等于-1，且不满足尺寸要求，则不算AP，否则是虚警
                    
        miss["imgs"][imgid] = {"objects":[]}
        wrong["imgs"][imgid] = {"objects":[]}
        right["imgs"][imgid] = {"objects":[]}
        miss_objs = miss["imgs"][imgid]["objects"]
        wrong_objs = wrong["imgs"][imgid]["objects"]
        right_objs = right["imgs"][imgid]["objects"]
        
        tt = 0
        for i in range(len(match_g)):
            if match_g[i] == -1:
                miss_objs.append(objs_g[i])                         # 记录漏报
        for i in range(len(match_r)):
            if match_r[i] == -1:
                obj = copy.deepcopy(objs_r[i])
                obj['correct_catelog'] = 'none'
                wrong_objs.append(obj)                              # 记录虚警
            elif match_r[i] != -2:                                  # 处理可以对应的情况。
                j = match_r[i]
                obj = copy.deepcopy(objs_r[i])
                if not check_type or objs_g[j]['category'] == objs_r[i]['category']:
                    right_objs.append(objs_r[i])
                    tt+=1
                else:
                    obj['correct_catelog'] = objs_g[j]['category']
                    wrong_objs.append(obj)
                    
        gt_num += len(objs_g) - match_g.count(-2)
        rc_n += len(objs_g) - match_g.count(-2)
        ac_n += len(objs_r) - match_r.count(-2)
        
        ac_c += tt
        rc_c += tt
    if types==None:
        styps = "all"
    elif len(types)==1:
        styps = list(types.keys())[0]
    elif not check_type or len(types)==0:
        styps = "none"
    else:
        styps = "[%s, ...total %s...]"%(list(types.keys())[0], len(types))
    report = "iou:%s, size:[%s,%s), types:%s, accuracy:%s, recall:%s"% (
        iou, minboxsize, maxboxsize, styps, 1 if ac_n==0 else ac_c*1.0/ac_n, 1 if rc_n==0 else rc_c*1.0/rc_n)
    summury = {
        "iou":iou,
        "accuracy":1 if ac_n==0 else ac_c*1.0/ac_n,
        "recall":1 if rc_n==0 else rc_c*1.0/rc_n,
        "miss":miss,
        "wrong":wrong,
        "right":right,
        "report":report
    }
    print(gt_num)
    return summury

with open(r'.\testjson\gt.json') as f:
    gt = json.loads(f.read())

with open(r'.\testjson\new.json') as f:
    result = json.loads(f.read())

type45 = "pn,pne,i5,p11,pl40,po,pl50,io,pl80,pl60,p26,i4,pl100,pl30,il60,i2,pl5,w57,p5,p10,pl120,il80,ip,p23,pr40,ph4.5,w59,p3,w55,pm20,p12,pg,pl70,pl20,pm55,il100,w13,p19,p27,ph4,pm30,wo,ph5,w32,p6"
type45 = type45.split(',')

# 判断目标时取消注释
# with open(r'.\testjson\gt_single_45.json') as f:
#     gt = json.loads(f.read())
# type45 = ['o']


with open('test_name_size.txt') as f:
    imgid = f.readlines()
imgid = list(map(lambda x:str(int(x.strip('\n').split(' ')[0])), imgid))

summary = eval_annos(gt, result, iou=0.5, imgids=imgid, check_type=True, types=type45, minscore=40, minboxsize=0, maxboxsize=32, match_same=True)
print('small recall:' + str(summary["recall"]))
print('small accuracy:' + str(summary["accuracy"]))

summary = eval_annos(gt, result, iou=0.5, imgids=imgid, check_type=True, types=type45, minscore=40, minboxsize=32, maxboxsize=96, match_same=True)
print('medium recall:' + str(summary["recall"]))
print('medium accuracy:' + str(summary["accuracy"]))

summary = eval_annos(gt, result, iou=0.5, imgids=imgid, check_type=True, types=type45, minscore=40, minboxsize=96, maxboxsize=400, match_same=True)
print('large recall:' + str(summary["recall"]))
print('large accuracy:' + str(summary["accuracy"]))
