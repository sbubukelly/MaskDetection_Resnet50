from PyQt5 import QtWidgets, QtGui, QtCore

from UI import Ui_MainWindow

import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import os
import sys
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import *
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import random
from Mask import Net


class MainWindow_controller(QtWidgets.QMainWindow):

    resnet = models.resnet50(pretrained=True)
    model = Net(resnet)

    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        self.ui.pushButton.clicked.connect(self.showModelStructure)
        self.ui.pushButton_3.clicked.connect(self.Result)

    def showModelStructure(self):
        print(self.model)

    def Result(self):
        classes = ('WithMask', 'WithoutMask')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torch.load('maskmodel.pth')
        model = model.to(device)
        model.eval()
        num = random.randrange(2)
        if num == 0:
            pathDir = os.listdir("./FaceMaskDataset/Test/WithMask")
        else:
            pathDir = os.listdir("./FaceMaskDataset/Test/WithoutMask")

        image = random.sample(pathDir, 1)
        if num == 0:
            img = cv2.imread("./FaceMaskDataset/Test/WithMask/" + image[0])
        else:
            img = cv2.imread("./FaceMaskDataset/Test/WithoutMask/" + image[0])

        # img = cv2.imread("./Wang_Leehom_closup.jpg")
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        trans = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        img = trans(img)
        img = img.to(device)
        img = img.unsqueeze(0)
        output = model(img)
        prob = F.softmax(output, dim=1)
        # print(prob)
        value, predicted = torch.max(output.data, 1)
        # print(predicted.item())
        # print(value)
        pred_class = classes[predicted.item()]
        print(pred_class)
        if num == 0:
            img = plt.imread("./FaceMaskDataset/Test/WithMask/" + image[0])
        else:
            img = plt.imread("./FaceMaskDataset/Test/WithoutMask/" + image[0])
        # img = plt.imread("./Wang_Leehom_closup.jpg")
        plt.imshow(img)
        plt.title("Class: "+pred_class)
        plt.show()
