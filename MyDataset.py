import torch
from torch.utils.data import DataLoader
import torchvision
'''
对于图像输入转换的settings
'''
kaggle_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

mnist_transform = torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])
kaggle_input_size = (3, 224, 224)
mnist_input_size = (1, 28, 28)
tang_input_size = (48,)
tang_vocab_size = 8293
'''
数据的路径，mnist只需要提供数据的上级文件夹路径
kaggle需要以文件夹分割类别图片，例如
/kaggle_train
    /kaggle_train/cat
    /kaggle_train/dog

/kaggle_test
    /kaggle_test/cat
    /kaggle_test/dog

/kaggle_pred
    /kaggle_pred/unknown
'''

mnist_root = './data/'
kaggle_test_path = './data/KAGGLE/toy_test/'
kaggle_pred_path = './data/KAGGLE/test/'
kaggle_train_path = './data/KAGGLE/toy_train/'
tang_train_path = './data/TANG/tang.npz'

def dataset_split(full_ds, train_rate=0.8):
    train_size = int(len(full_ds) * train_rate)
    validate_size = len(full_ds) - train_size
    train_ds, validate_ds = torch.utils.data.random_split(full_ds, [train_size, validate_size])
    return train_ds, validate_ds


def get_mnist_train_dataloader(batch_size_train=64, rate=0.8):

    dataset_train = torchvision.datasets.MNIST(mnist_root, train=True, download=False,
                                   transform=mnist_transform)

    dataset_train, dataset_val = dataset_split(dataset_train, rate)
    train_loader = DataLoader(dataset_train, batch_size=batch_size_train, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size_train, shuffle=False)

    return train_loader, val_loader

def get_mnist_test_dataloader(batch_size_test=64):
    test_loader = DataLoader(
        torchvision.datasets.MNIST(mnist_root, train=False, download=False,
                                   transform=mnist_transform),
        batch_size=batch_size_test, shuffle=False)

    return test_loader


def get_kaggle_train_dataloader(batch_size_train=64, rate=0.8):

    dataset_train = torchvision.datasets.ImageFolder(kaggle_train_path, kaggle_transform)

    dataset_train, dataset_val = dataset_split(dataset_train, rate)

    train_loader = torch.utils.data.DataLoader(dataset_train,
                                               batch_size=batch_size_train, shuffle=True)

    val_loader = torch.utils.data.DataLoader(dataset_val,
                                                 batch_size=batch_size_train, shuffle=False)

    return train_loader, val_loader


def get_kaggle_dataloader(batch_size_test=64, path=kaggle_test_path):

    dataset_test = torchvision.datasets.ImageFolder(path, kaggle_transform)

    test_loader = torch.utils.data.DataLoader(dataset_test,
                                              batch_size=batch_size_test, shuffle=False)

    return test_loader


import numpy as np


class PoemDataSet(torch.utils.data.Dataset):
    def __init__(self,poem_path,seq_len):
        self.seq_len = seq_len
        self.poem_path = poem_path
        self.poem_data, self.ix2word, self.word2ix = get_tang_raw_data(poem_path)
        self.no_space_data = self.filter_space()

    def __getitem__(self, idx:int):
        txt = self.no_space_data[idx*self.seq_len : (idx+1)*self.seq_len]
        label = self.no_space_data[idx*self.seq_len + 1 : (idx+1)*self.seq_len + 1] # 将窗口向后移动一个字符就是标签
        txt = torch.from_numpy(np.array(txt)).long()
        label = torch.from_numpy(np.array(label)).long()
        return txt,label

    def __len__(self):
        return int(len(self.no_space_data) / self.seq_len)

    def filter_space(self): # 将空格的数据给过滤掉，并将原始数据平整到一维
        t_data = torch.from_numpy(self.poem_data).view(-1)
        flat_data = t_data.numpy()
        no_space_data = []
        for i in flat_data:
            if (i != 8292 ):
                no_space_data.append(i)
        return no_space_data




def get_tang_train_dataloader(batch_size=16, rate=0.8):
    dataset_train = PoemDataSet(tang_train_path, tang_input_size[0])
    dataset_train, dataset_val = dataset_split(dataset_train, rate)
    train_dataloader = torch.utils.data.DataLoader(dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2)
    val_dataloader = torch.utils.data.DataLoader(dataset_val,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=2)

    return train_dataloader, val_dataloader



def get_tang_raw_data(poem_path):
    datas = np.load(poem_path,allow_pickle=True)  #numpy 1.16.2  以上引入了allow_pickle
    #datas = np.load(self.poem_path)
    data = datas['data']
    ix2word = datas['ix2word'].item()
    word2ix = datas['word2ix'].item()
    return data, ix2word, word2ix


