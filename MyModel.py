import torch.nn as nn
import numpy as np
import pandas as pd

class MnistCNN(nn.Module):
    def __init__(self):
        super(MnistCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            #nn.BatchNorm2d(16)
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10)
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.out(x)
        return x


class KaggleCNN(nn.Module):
    def __init__ (self):
        super(KaggleCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3),

            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)

        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 7),
            nn.ReLU(),
            nn.MaxPool2d(2)

        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 9),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)

        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        #input (bs, 3, 150, 150)
        x = self.conv1(x) #input (bs, 32, 111, 111)
        x = self.conv2(x) #input (bs, 64, 53, 53)
        x = self.conv3(x)#input (bs, 64, 23, 23)
        x = self.conv4(x)#input (bs, 64, 7, 7)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

from torchvision import models



class NetFactory:
    def getNet(self, mode='mnist', out_dim=2):
        if mode == 'mnist':
            return MnistCNN()
        elif mode == 'kaggle':
            return KaggleCNN()

        elif mode == 'resnet18':
            resnet18 = models.resnet18(pretrained=True)
            resnet18.fc = nn.Linear(512, out_dim)  # 将最后一层全连接的输出调整到2维
            return resnet18
        elif mode == 'resnet50':
            resnet50 = models.resnet50(pretrained=True)
            resnet50.fc = nn.Linear(2048, out_dim)  # 将最后一层全连接的输出调整到2维
            return resnet50
        else:
            raise KeyError("No exist model!")




