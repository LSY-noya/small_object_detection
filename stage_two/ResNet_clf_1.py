import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
import argparse

#region 1 超参数
parser = argparse.ArgumentParser(description = 'Generate a new dataset')
parser.add_argument('-d', '--data', help = 'dataset dir')
parser.add_argument('-b', '--bs', help = 'batch size', type=int, default=16)
parser.add_argument('-l', '--lr', help = 'learning rate', type=float, default=0.0001)
parser.add_argument('-e', '--epoch', help = 'epoch number', type=int, default=500)
parser.add_argument('-m', '--model', help = 'model name')
args = parser.parse_args()

BATCH_SIZE = args.bs
LEARNING_RATE = args.lr
EPOCH = args.epoch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#endregion

#region 2 类别
type45 = "pn,pne,i5,p11,pl40,po,pl50,io,pl80,pl60,p26,i4,pl100,pl30,il60,i2,pl5,w57,p5,p10,pl120,il80,ip,p23,pr40,ph4.5,w59,p3,w55,pm20,p12,pg,pl70,pl20,pm55,il100,w13,p19,p27,ph4,pm30,wo,ph5,w32,p6,pb,wb,ib,z"
type45 = type45.split(',')

typenum = [1832, 1339, 1016, 944, 842, 691, 635, 1698, 1727, 1562, 1529, 1409, 1347, 1480, 1355, 1179, 1059, 1033, 1026, 966, 832, 1182, 1131, 971, 814, 719, 725, 666, 650, 640, 634, 616, 612, 581, 587, 552, 534, 516, 499, 491, 449, 428, 437, 406, 411, 648, 186, 164, 15542]
typenum = np.array(typenum)
#endregion

#region 3 定义TT100k的dataset
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
        img = cv2.resize(img,(96, 96),interpolation=cv2.INTER_LINEAR)
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
#endregion

#region 3.5 focal loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        return

    def forward(self, predict, label):
        label = label.view(-1, 1)
        pred = F.softmax(predict, dim=1)
        pt = pred.gather(1, label)
        pt = pt.view(-1, 1)
        log_pt = torch.log(pt)

        loss = -1 * ((1 - pt) ** self.gamma) * log_pt

        return loss.sum()
#endregion

#region 3.5 ClassBalanced loss
class CBLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, beta=0.999):
        super(CBLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.weight = (1 - self.beta) / (1 - self.beta ** typenum)
        return

    def forward(self, predict, label):
        label = label.view(-1, 1)
        pred = F.softmax(predict, dim=1)
        pt = pred.gather(1, label)
        pt = pt.view(-1, 1)
        log_pt = torch.log(pt)

        variant = torch.Tensor(self.weight[label.cpu()])
        variant = variant.view(-1, 1)
        variant = variant.to(DEVICE)
        loss = -100 * variant * log_pt
        
        return loss.sum()
#endregion

#region 4 定义模型
model = models.resnet18(pretrained=True)
fc_features = model.fc.in_features
model.fc = nn.Linear(fc_features, 49)

# model = models.vgg11(pretrained=True)
# model.classifier = nn.Sequential(nn.Linear(512 * 7 * 7,4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096,4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, 49))

# model = models.mobilenet_v2(pretrained=True)
# model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.last_channel, 49))

model = model.to(DEVICE)
# criterion = nn.CrossEntropyLoss(size_average=False)  # logsoftmax函数
# criterion = FocalLoss()
criterion = CBLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE,momentum=0.9)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[80,160],gamma = 0.1)
#endregion

#region 5 train
train_data = TT100K_Dataset(root=args.data, data_txt='trainval_result_0p08_3.txt', mode=1)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if(batch_idx + 1) % 30 == 0: 
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

for epoch in range(1, EPOCH + 1):
    train(epoch)
    scheduler.step()
    if epoch % 50 ==0:
        torch.save(model.state_dict(), args.model + '_%d.pt' % epoch)

#endregion

#region 6 test
test_data = TT100K_Dataset(root=args.data, data_txt='test_result.txt', mode=0)
test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

def test():
    model.eval()
    for batch_idx, data in enumerate(test_loader):
        data = data.to(DEVICE)
        output = model(data)
        pred = output.max(1, keepdim=True)[1] # 找到概率最大的下标
        pred = pred.cpu()
        pred = pred.numpy()
        with open('test_pred_res18_LINEAR_CB.txt','a') as f:
            for m in pred:
                f.write(type45[m[0]] + '\n')
test()
#endregion
