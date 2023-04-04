import torch
from torch.utils.data import DataLoader
import torchvision


def get_mnist_dataloader(batch_size_train=64, batch_size_test=128):
    train_loader = DataLoader(
        torchvision.datasets.MNIST('/mnt/c/Code/DL23spring/data/', train=True, download=False,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_train, shuffle=True)
    test_loader = DataLoader(
        torchvision.datasets.MNIST('/mnt/c/Code/DL23spring/data/', train=False, download=False,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_test, shuffle=False)

    return train_loader, test_loader

def get_kaggle_dataloader(batch_size_train=64, batch_size_test=128):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(150),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    dataset_train = torchvision.datasets.ImageFolder(root + '/train', transform)
    dataset_test = torchvision.datasets.ImageFolder(root + '/val', transform)
    train_loader = torch.utils.data.DataLoader(dataset_train,
                                               batch_size=batch_size_train, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset_test,
                                              batch_size=batch_size_test, shuffle=False)
    return train_loader, test_loader