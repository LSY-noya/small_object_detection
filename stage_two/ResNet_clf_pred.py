import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
import argparse

type45 = "pn,pne,i5,p11,pl40,po,pl50,io,pl80,pl60,p26,i4,pl100,pl30,il60,i2,pl5,w57,p5,p10,pl120,il80,ip,p23,pr40,ph4.5,w59,p3,w55,pm20,p12,pg,pl70,pl20,pm55,il100,w13,p19,p27,ph4,pm30,wo,ph5,w32,p6,pb,wb,ib,z"
type45 = type45.split(',')

parser = argparse.ArgumentParser(description = 'Generate a new dataset')
parser.add_argument('-d', '--data', help = 'dataset dir')
parser.add_argument('-m', '--model', help = 'model path')
args = parser.parse_args()

class TT100K_Dataset(Dataset):
    def __init__(self, root, data_txt, mode):
        with open(os.path.join(root, data_txt)) as f:
            data = f.readlines()
        data = list(map(lambda x:x.strip('\n').split(' '), data))
        self.mode = mode
        if mode:
            # self.imgs_path = [m[0] for m in data]
            self.imgs_path = [os.path.join(root, m[0]) for m in data]
        else:
            self.imgs_path = [os.path.join(root, m[0]) for m in data]
        self.labels = [m[1] for m in data]
        self.mean = [0., 0., 0.]
        self.std = [0., 0., 0.]
        if os.path.exists('mean_std_3.txt'):
            with open('mean_std_3.txt') as f:
                mean_std = f.readlines()
            mean_std = list(map(lambda x:x.strip('\n').split(' '), mean_std))
            self.mean = [float(mean_std[0][0]), float(mean_std[0][1]), float(mean_std[0][2])]
            self.std = [float(mean_std[1][0]), float(mean_std[1][1]), float(mean_std[1][2])]
            print("normMean = {}".format(self.mean))
            print("normStd = {}".format(self.std))
        else:
            self.get_mean_std()
            self.mean = list(self.mean)
            self.std = list(self.std)
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=self.mean, std=self.std)]) 

    def __getitem__(self, index):
        # 处理图片
        img_path = self.imgs_path[index]
        img = cv2.imread(img_path)
        img = cv2.resize(img,(96, 96),interpolation=cv2.INTER_CUBIC)
        img = self.transform(img)
        # 处理标签
        if self.mode :
            label = type45.index(self.labels[index])
            return img, label
        else:
            return img
 
    def __len__(self):
        return len(self.imgs_path)
    
    def get_mean_std(self):
        num_imgs = len(self.imgs_path)
        for path in self.imgs_path:
            img = cv2.imread(path)
            for i in range(3):
                self.mean[i] += img[i, :, :].mean()
                self.std[i] += img[i, :, :].std()
        self.mean = np.array(self.mean) / (num_imgs * 255)
        self.std = np.array(self.std) / (num_imgs * 255)
        with open('mean_std_3.txt', 'a') as f:
            f.write("{} {} {}\n".format(self.mean[0], self.mean[1], self.mean[2]))
            f.write("{} {} {}\n".format(self.std[0], self.std[1], self.std[2]))
        print("normMean = {}".format(self.mean))
        print("normStd = {}".format(self.std))
        return

test_data = TT100K_Dataset(root=args.data, data_txt='test_result.txt', mode=0)
test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = models.resnet18(pretrained=False, num_classes=49)
model = models.resnet50(pretrained=False, num_classes=49)
model.load_state_dict(torch.load(args.model))
model = model.to(DEVICE)

def test():
    model.eval()
    for batch_idx, data in enumerate(test_loader):
        data = data.to(DEVICE)
        output = model(data)
        pred = output.max(1, keepdim=True)[1] # 找到概率最大的下标
        pred = pred.cpu()
        pred = pred.numpy()
        with open('test_pred.txt','a') as f:
            for m in pred:
                f.write(type45[m[0]] + '\n')
test()
