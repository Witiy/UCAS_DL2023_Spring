import os

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import seaborn as sns
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt
class Estimator:
    def __init__(self, model: torch.nn.Module, input_size, device, metrics=None):


        self.early_dict = None
        self.optim = None
        self.val_dataloader = None
        self.train_dataloader = None
        self.loss = None
        self.model = model

        self.device = device
        self.model.to(self.device)
        if metrics is None:
            self.added_metrics = ['accuracy']
        else:
            self.added_metrics = metrics
        #summary(model, input_size, -1)
        self.smy(model, input_size)

    def smy(self, model, input_size):
        summary(model, input_size, -1)

    def prepare_test(self, test_dataloader, device=None, loss=None, metrics=None,
                      ):

        self.test_dataloader = test_dataloader
        print('test dataset size: {}'.format(len(test_dataloader.dataset)))
        self.device = device

        if loss == None:
            self.loss = torch.nn.CrossEntropyLoss()
        elif loss == 'ce':
            self.loss = torch.nn.CrossEntropyLoss()
        elif loss == 'bce':
            self.loss = torch.nn.BCEWithLogitsLoss()
        else:
            raise KeyError

        self.loss.to(self.device)

        if metrics != metrics:
            self.added_metrics = metrics


    def prepare_train(self, train_dataloader, val_dataloader=None,
                      device=None, lr=1e-3, optim=None, loss=None, early_stopping=True, early_dict=None, metrics=None,
                      ):


        self.early_stopping = early_stopping
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        if val_dataloader:
            print('train dataset size: {}, val dataset size: {}'.format(len(train_dataloader.dataset),
                                                                     len(val_dataloader.dataset)))
        else:
            print('train dataset size: {}'.format(len(train_dataloader.dataset)))
        self.device = device

        if optim == None:
            self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        else:
            self.optim = optim

        if loss == None:
            self.loss = torch.nn.CrossEntropyLoss()
        elif loss == 'ce':
            self.loss = torch.nn.CrossEntropyLoss()
        elif loss == 'bce':
            self.loss = torch.nn.BCEWithLogitsLoss()
        else:
            raise KeyError

        self.loss.to(self.device)

        self.early_dict = {
            'patient': 5,
            'metrics': 'loss',
            'mode': 'min'
        }
        if early_dict != None:
            for key in early_dict.keys():
                self.early_dict[key] = early_dict[key]

        self.stop = False  # for early stopping state
        self.best_matrics = None
        self.best_model_para = None
        self.patient = 0
        self.history = {'train_loss': [], 'val_loss': []
                        }
        if metrics != metrics:
            self.added_metrics = metrics
        self.init_history()

    def init_history(self):
        for m in self.added_metrics:
            self.history['train_' + m] = []
            self.history['val_' + m] = []



    def optimize(self, i):

        self.model.train()
        loop = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader))
        train_loss = 0.
        num = 0
        value = {}
        for metrics in self.added_metrics:
            value[metrics] = 0.

        for batch_idx, item in loop:
                loop.set_description('Epoch %i' % i)
                self.optim.zero_grad()

                loss, output, target = self.calcu_loss(item)
                loss.backward()
                self.optim.step()

                with torch.no_grad():
                    train_loss += loss.item()
                    num += target.shape[0]
                    postfix = {}
                    for metrics in self.added_metrics:
                        value[metrics] += self.calcu_metrics(metrics, output, target).item()#记录训练过程中所要求的metrics, e.g. accuracy
                        #print(value[metrics], num)
                        postfix['train_'+metrics] = ' {:.6f}'.format(value[metrics] / num)
                    postfix['train_loss'] = ' {:.6f}'.format(train_loss / (batch_idx + 1))
                    loop.set_postfix(postfix)

        self.history['train_loss'].append(train_loss / len(self.train_dataloader)) #记录训练过程的loss
        for metrics in self.added_metrics:
            value[metrics] /= num #对记录的metrics求平均
            self.history['train_' + metrics].append(value[metrics])


    def check(self):
        #获得当前epoch结束后，模型在验证集上的各项指标
        current_metrics = self.history['val_' + self.early_dict['metrics']][-1]

        #若为第一个epoch的处理
        if self.best_matrics == None:
            self.best_matrics = current_metrics
            self.best_model_para = self.model.state_dict()
            self.stop = False
            return
        #计算当前epoch训练后，性能是否提升，对于accuracy来说，mode=max，对于loss来说，mode=min
        if self.early_dict['mode'] == 'min':
            improve = self.best_matrics - current_metrics
        else:
            improve = current_metrics - self.best_matrics
        #若有所提升，则记录为到目前为止的最佳性能，同时暂存模型参数
        if improve > 0:
            self.best_matrics = current_metrics
            self.best_model_para = self.model.state_dict()
            torch.save(self.best_model_para, 'tmp_model.pth')
            self.patient = 0
            self.end_epoch = len(self.history['val_' + self.early_dict['metrics']])
        #若无提升，则记录无提升epoch数+1
        else:
            self.patient += 1
        #若超过规定无提升epoch数，停止训练
        if self.patient > self.early_dict['patient']:
            self.stop = True
        else:
            self.stop = False

    def preprocess_train(self):
        pass

    def postprocess_epoch(self):
        pass

    def train(self, epoch=50):
        self.preprocess_train()
        self.init_history()
        self.eval('train')
        if self.val_dataloader != None:
            self.eval('val')

        for i in range(epoch):
            self.optimize(i)
            if self.val_dataloader != None:
                self.eval('val')

            if self.early_stopping:
                self.check()
                if self.stop:
                    break
            self.postprocess_epoch()
        print('Finish Train')

    def calcu_loss(self, item):
        data, target = item[0], item[1]
        data, target = data.to(self.device), target.to(self.device)

        output = self.model(data)
        return self.loss(output, target), output, target



    def calcu_metrics(self, metric, output, target):
        return getattr(self, metric)(output, target)

    ## topk的准确率计算
    def accuracy(self, output, target, topk=1):

        maxk = topk
        # 获取前K的索引
        _, pred = output.topk(maxk, 1, True, True)  # 使用topk来获得前k个的索引
        pred = pred.t()  # 进行转置

        correct = pred.eq(target.view(1, -1).expand_as(pred))  # 与正确标签序列形成的矩阵相比，生成True/False矩阵
        return correct.float().sum()



    def eval(self, mode='val'):
        self.model.eval()
        test_loss = 0

        value = {}
        for metrics in self.added_metrics:
            value[metrics] = 0.

        if mode == 'val':
            dataloader = self.val_dataloader
        elif mode == 'train':
            dataloader = self.train_dataloader
        elif mode == 'test':
            dataloader = self.test_dataloader

        else:
            raise KeyError

        with torch.no_grad():

            loop = tqdm(enumerate(dataloader), total=len(dataloader))
            num = 0

            for batch_idx, item in loop:

                loop.set_description('Eval('+mode+')')
                loss, output, target = self.calcu_loss(item)
                test_loss += loss.item()
                num += target.shape[0]
                postfix = {}
                for metrics in self.added_metrics:
                    value[metrics] += self.calcu_metrics(metrics, output, target).item()
                    #print(value[metrics], num)
                    postfix[mode + '_' + metrics] = ' {:.6f}'.format(value[metrics] / num)
                postfix[mode + '_' + 'loss'] = ' {:.6f}'.format(test_loss / (batch_idx + 1))
                loop.set_postfix(postfix)


        test_loss /= len(dataloader)
        if mode != 'test':
            self.history[mode + '_loss'].append(test_loss)
        for metrics in self.added_metrics:
            value[metrics] /= num
            if mode != 'test':
                self.history[mode + '_' + metrics].append(value[metrics])




    def save(self, path):
        if self.early_stopping:
            os.system('mv tmp_model.pth '+path)
        else:
            torch.save(self.model.state_dict(), path)
        np.save(path + '.history.npy', self.history)



    def plot_history(self, metrics='loss', saved=True):
        train_m = 'train_' + metrics
        test_m = 'val_' + metrics
        sep = int(len(self.history[train_m]) / len(self.history[test_m]))

        sns.lineplot(x=range(0, len(self.history[train_m]), sep), y=self.history[train_m], label=train_m)
        sns.lineplot(x=range(len(self.history[test_m])), y=self.history[test_m], label=test_m)
        if self.early_stopping:
            plt.axvline(x=self.end_epoch - 1, linestyle='--', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        if saved:
            plt.savefig('history_'+metrics+'.png')
        plt.close()