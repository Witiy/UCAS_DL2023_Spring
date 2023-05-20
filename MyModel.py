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
            nn.BatchNorm2d(32),
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
            nn.BatchNorm2d(64),
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

class TangModel(nn.Module):
    def __init__(self, vocab_size):
        super(TangModel, self).__init__()
        self.hidden_dim = 64
        self.embedding_dim = 64
        self.num_layers = 3
        self.embeddings = nn.Embedding(vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True, dropout=0.3)
        self.lkrelu = nn.LeakyReLU()
        self.linear1 = nn.Linear(self.hidden_dim, 1024)
        self.drop1 = nn.Dropout(0.2)
        self.linear2 = nn.Linear(1024, 4096)
        self.drop2 = nn.Dropout(0.4)
        self.linear3 = nn.Linear(4096, vocab_size)

    def forward(self, input, hidden=None):
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        embeds = self.embeddings(input.long())
        # [batch, seq_len] => [batch, seq_len, embed_dim]

        if hidden is None:
            h_0 = input.data.new(self.num_layers, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(self.num_layers, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden
        output, hidden = self.lstm(embeds, (h_0, c_0))

        output = self.lkrelu(self.drop1(self.linear1(output)))

        output = self.lkrelu(self.drop2(self.linear2(output)))

        output = self.linear3(output)

        output = output.reshape(batch_size * seq_len, -1)
        return output, hidden





from torchvision import models



class NetFactory:
    def getNet(self, mode='mnist', **kwargs):
        if mode == 'mnist':
            return MnistCNN()
        elif mode == 'kaggle':
            return KaggleCNN()
        elif mode == 'tang':
            return TangModel(kwargs['vocab_size'])

        elif mode == 'resnet18':
            resnet18 = models.resnet18(pretrained=True)
            resnet18.fc = nn.Linear(512, 2)  # 将最后一层全连接的输出调整到2维
            return resnet18
        elif mode == 'resnet50':
            resnet50 = models.resnet50(pretrained=True)
            resnet50.fc = nn.Linear(2048, 2)  # 将最后一层全连接的输出调整到2维
            return resnet50
        else:
            raise KeyError("No exist model!")




