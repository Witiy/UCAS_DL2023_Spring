import torch
import argparse
from MyModel import NetFactory
from MyDataset import get_mnist_dataloader
from MyEstimator import Estimator

def call_mnist(mode, path, cuda, lr, bs, es, early_stopping, early_dict):


    factory = NetFactory()
    model = factory.getNet('mnist')
    if cuda:
        device = torch.device("cuda:0")
        model.to(device)
    else:
        device = torch.device("cpu")
        model.to(device)

    print('loading data..')
    train_dataloader, test_dataloader = get_mnist_dataloader(batch_size_train=bs, batch_size_test=2*bs)
    estimator = Estimator(model, train_dataloader, test_dataloader, device,
        lr= lr, early_stopping=early_stopping, early_dict=early_dict,
        )

    if mode == 'test':

        estimator.model.load_state_dict(torch.load(path))
        print('start testing..')
        estimator.eval('test', test_dataloader=test_dataloader, verbose=1)
    else:
        print('start training..')
        estimator.train(es)
        estimator.save(path)
        #estimator.plot_history()
    return estimator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='mnist')
    parser.add_argument('--path', type=str, default='./model/MNIST/model.pth')
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--es', type=int, default=10)
    parser.add_argument('--early_stopping', action='store_true', default=True)
    parser.add_argument('--patient', type=int, default=5)
    parser.add_argument('--cuda', action='store_true', default=True)

    args = parser.parse_args()
    mode, path, cuda, lr, bs, es = args.mode, args.path, args.cuda, args.lr, args.bs, args.es

    early_stopping = args.early_stopping
    if early_stopping:
        early_dict = {
            'patient':args.patient
        }
    else:
        early_dict = None

    if args.experiment == 'mnist':
        call_mnist(mode, path, cuda, lr, bs, es, early_stopping, early_dict)

if __name__ == '__main__':
    main()

