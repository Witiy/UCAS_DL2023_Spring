import torch
import argparse
from MyModel import NetFactory
from MyDataset import *
from MyEstimator import Estimator
import numpy as np
import pandas as pd

mnist_model_path = '/mnt/c/Code/DL23Spring/model/MNIST/model.pth'
kaggle_model_path = '/mnt/c/Code/DL23Spring/model/KAGGLE/model.pth'
kaggle_output_path = '/mnt/c/Code/DL23Spring/model/KAGGLE/output.csv'

def call_mnist(mode, path=mnist_model_path, cuda=True, lr=1e3, bs=256, es=20, early_stopping=True, early_dict=None, tr=0.8):
    if path == None:
        path = mnist_model_path
    if cuda == None:
        cuda = True
    if lr == None:
        lr = 1e-3
    if bs == None:
        bs = 256
    if es == None:
        es = 20
    if early_stopping == None:
        early_stopping = True
    if tr == None:
        tr = 0.8
    factory = NetFactory()
    model = factory.getNet('mnist')
    if cuda:
        device = torch.device("cuda:0")
        model.to(device)
    else:
        device = torch.device("cpu")
        model.to(device)

    print('loading data..')

    estimator = Estimator(model, mnist_input_size, device)

    if mode == 'test':
        test_dataloader = get_mnist_test_dataloader(bs)
        estimator.prepare_test(test_dataloader, device=device)
        estimator.model.load_state_dict(torch.load(path))
        print('start testing..')
        estimator.eval('test')
    else:
        train_dataloader, val_dataloader = get_mnist_train_dataloader(bs, tr)
        estimator.prepare_train(train_dataloader, val_dataloader, device, lr, early_stopping=early_stopping, early_dict=early_dict)
        print('start training..')
        estimator.train(es)
        estimator.save(path)
        estimator.plot_history('loss')
        estimator.plot_history('acc')
    return estimator


def call_kaggle(mode, path=kaggle_model_path, cuda=True, lr=1e3, bs=256, es=20, early_stopping=True, early_dict=None, tr=0.8, out_path=kaggle_output_path, arch=''):
    if path == None:
        path = kaggle_model_path
    if cuda == None:
        cuda = True
    if lr == None:
        lr = 1e-3
    if bs == None:
        bs = 128
    if es == None:
        es = 20
    if early_stopping == None:
        early_stopping = True
    if tr == None:
        tr = 0.8
    if out_path == None:
        out_path = kaggle_output_path
    if arch == None:
        arch = 'resnet18'

    factory = NetFactory()
    if arch == '':
        model = factory.getNet('kaggle')
    else:
        model = factory.getNet(arch, 2)

    if cuda:
        device = torch.device("cuda:0")
        model.to(device)
    else:
        device = torch.device("cpu")
        model.to(device)

    print('loading data..')


    estimator = Estimator(model, kaggle_input_size, device)

    if mode == 'test':
        test_dataloader = get_kaggle_dataloader(bs, kaggle_test_path)
        estimator.prepare_test(test_dataloader, device=device)
        estimator.model.load_state_dict(torch.load(path))
        print('start testing..')
        estimator.eval('test')

    elif mode == 'pred':
        pred_dataloader = get_kaggle_dataloader(bs, kaggle_pred_path)
        estimator.model.load_state_dict(torch.load(path))
        print('start predicting..')
        outputs = estimator.pred(test_dataloader=pred_dataloader)
        outputs = outputs.data.max(1, keepdim=True)[1]
        ids = list(range(1, len(outputs)+1))
        output_df = pd.DataFrame({'id':ids, 'label':outputs})
        output_df.to_csv(out_path, index=False)

    else:
        train_dataloader, val_dataloader = get_kaggle_train_dataloader(batch_size_train=bs, rate=tr)
        estimator.prepare_train(train_dataloader, val_dataloader, device,
            lr= lr, early_stopping=early_stopping, early_dict=early_dict
        )
        print('start training..')
        estimator.train(es)
        estimator.save(path)
        #estimator.plot_history()
    return estimator



def main():
    parser = argparse.ArgumentParser()
    #optional mnist kaggle
    parser.add_argument('--experiment', type=str, default='kaggle')
    parser.add_argument('--path', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--lr', type=float)
    parser.add_argument('--bs', type=int)
    parser.add_argument('--es', type=int, default=20)
    parser.add_argument('--tr', type=float)
    parser.add_argument('--net', type=str)
    parser.add_argument('--early_stopping', action='store_true', default=True)

    parser.add_argument('--patient', type=int, default=5)
    parser.add_argument('--cuda', action='store_true', default=True)

    args = parser.parse_args()
    mode, output, path, cuda, lr, bs, es, tr = args.mode, args.output, args.path, args.cuda, args.lr, args.bs, args.es, args.tr

    early_stopping = args.early_stopping
    if early_stopping:
        early_dict = {
            'patient':args.patient
        }
    else:
        early_dict = None

    if args.experiment == 'mnist':
        call_mnist(mode, path, cuda, lr, bs, es, early_stopping, early_dict, tr)
    elif args.experiment == 'kaggle':
        call_kaggle(mode, path, cuda, lr, bs, es, early_stopping, early_dict, tr, output, args.net)

if __name__ == '__main__':
    main()

