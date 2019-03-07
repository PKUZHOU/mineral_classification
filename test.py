import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torch.utils.data import DataLoader
from data import MyDataset
import argparse
import cv2
from network import resnet18
import os
import numpy as np
import torch.nn.functional as f

net = torch.load("net.pkl")
net = net.cuda()

val_path = "/home/zhou/mineral_data/val"
files = os.listdir(val_path)
for image in files:
    path = os.path.join(val_path,image)
    cv_image = cv2.imread(path)
    input = torch.from_numpy(cv_image.astype(np.float32)).permute(2,0,1)
    input = (input-127.5)/127.5
    input = input.unsqueeze(0)
    input = Variable(input).cuda()
    pred = net(input)
    pred = f.softmax(pred,dim=1)
    pred_kind = torch.argmax(pred)
    print pred_kind.cpu().data
    print "prob ",pred[0][pred_kind].cpu().data
    cv2.imshow("test",cv_image)
    cv2.waitKey(0)