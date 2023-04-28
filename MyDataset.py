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

mnist_root = '/mnt/c/Code/DL23spring/data/'
kaggle_test_path = '/mnt/c/Code/DL23spring/data/KAGGLE/toy_test/'
kaggle_pred_path = '/mnt/c/Code/DL23spring/data/KAGGLE/test/'
kaggle_train_path = '/mnt/c/Code/DL23spring/data/KAGGLE/toy_train/'


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




