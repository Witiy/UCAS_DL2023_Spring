import torch
import argparse

import MyDataset
from MyModel import NetFactory
from MyDataset import *
from MyEstimator import Estimator
import numpy as np
import pandas as pd

mnist_model_path = './model/MNIST/model.pth'
kaggle_model_path = './model/KAGGLE/model.pth'
tang_model_path = './model/TANG/model.pth'
kaggle_output_path = './model/KAGGLE/output.csv'

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


def call_kaggle(mode, path=kaggle_model_path, cuda=True, lr=5e4, bs=128, es=20, early_stopping=True, early_dict=None, tr=0.8, arch='default', out_path=kaggle_output_path):
    if path == None:
        path = kaggle_model_path
    if cuda == None:
        cuda = True
    if lr == None:
        lr = 5e-4
    if bs == None:
        bs = 128
    if es == None:
        es = 50
    if early_stopping == None:
        early_stopping = True
    if tr == None:
        tr = 0.8
    if out_path == None:
        out_path = kaggle_output_path
    if arch == None:
        arch = 'resnet18'

    factory = NetFactory()
    if arch == 'default':
        model = factory.getNet('kaggle')
    else:
        path = path[:-4] + '_' + arch + '.pth'
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
        output_df = pd.DataFrame({'id':ids, 'label':outputs.cpu().numpy()[:,0]})
        output_df.to_csv(out_path, index=False)

    else:
        train_dataloader, val_dataloader = get_kaggle_train_dataloader(batch_size_train=bs, rate=tr)
        estimator.prepare_train(train_dataloader, val_dataloader, device,
            lr= lr, early_stopping=early_stopping, early_dict=early_dict
        )
        print('start training..')
        estimator.train(es)
        estimator.save(path)

        estimator.plot_history('loss')
        estimator.plot_history('acc')

    return estimator



from LangEstimator import langEstimator
def call_tang(mode, path=None, cuda=None, lr=None, bs=None, es=None, early_stopping=True, early_dict=None, tr=None,):
    if path == None:
        path = tang_model_path
    if cuda == None:
        cuda = True
    if lr == None:
        lr=5e-4
    if bs == None:
        bs = 16
    if es == None:
        es = 50
    if tr == None:
        tr = 0.8

    factory = NetFactory()

    model = factory.getNet('tang', vocab_size=tang_vocab_size)

    if cuda:
        device = torch.device("cuda:0")
        model.to(device)
    else:
        device = torch.device("cpu")
        model.to(device)

    print('loading data..')


    estimator = langEstimator(model, tang_input_size, device)

    if mode == 'train':
        train_dataloader, val_dataloader = get_tang_train_dataloader(batch_size=bs, rate=tr)
        estimator.prepare_train(train_dataloader, val_dataloader, device=device,
                                lr=lr, early_stopping=early_stopping, early_dict=early_dict, loss='ce'
                                )
        print('start training..')
        estimator.train(es)
        estimator.save(path)

        estimator.plot_history('loss')
        estimator.plot_history('acc')

    elif mode == 'gen':
        _, ix2word, word2ix = get_tang_raw_data(MyDataset.tang_train_path)
        estimator.model.load_state_dict(torch.load(path))
        print('start predicting..')
        outputs = estimator.generate('雨', ix2word,word2ix,tang_input_size[0])
        print(outputs)





    return estimator




def main():
    parser = argparse.ArgumentParser()
    #optional mnist kaggle
    parser.add_argument('--experiment', type=str, default='tang')
    parser.add_argument('--path', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--lr', type=float)
    parser.add_argument('--bs', type=int)
    parser.add_argument('--es', type=int,)
    parser.add_argument('--tr', type=float)
    parser.add_argument('--net', type=str, default='default')
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
        call_kaggle(mode, path, cuda, lr, bs, es, early_stopping, early_dict, tr, args.net, output)
    elif args.experiment == 'tang':
        call_tang(mode, path, cuda, lr, bs, es, early_stopping, early_dict, tr)

if __name__ == '__main__':
    main()

