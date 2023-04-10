import torch
from tqdm import tqdm
import seaborn as sns
from torchsummary import summary


class Estimator:
    def __init__(self, model: torch.nn.Module, input_size):
        self.early_dict = None
        self.optim = None
        self.val_dataloader = None
        self.train_dataloader = None
        self.loss = None
        self.model = model
        summary(model, input_size, -1)

    def prepare_train(self, train_dataloader, val_dataloader,
                      device=None, lr=1e-3, optim=None, loss=None, early_stopping=True, early_dict=None, metrics=None,
                      ):

        if metrics is None:
            metrics = ['acc']
        self.early_stopping = early_stopping
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        print('train dataset size: {}, test dataset size: {}'.format(len(train_dataloader.dataset),
                                                                     len(val_dataloader.dataset)))
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

        self.added_metrics = metrics
        self.init_history()

    def init_history(self):
        for m in self.added_metrics:
            self.history['train_' + m] = []
            self.history['val_' + m] = []

    def optimize(self, ):
        self.model.train()
        for batch_idx, (data, target) in tqdm(enumerate(self.train_dataloader)):
            data, target = data.to(self.device), target.to(self.device)
            self.optim.zero_grad()
            output = self.model(data)
            loss = self.loss(output, target)

            loss.backward()
            self.optim.step()

    def check(self):
        current_metrics = self.history['val_' + self.early_dict['metrics']][-1]
        if self.best_matrics == None:
            self.best_matrics = current_metrics
            self.best_model_para = self.model.state_dict()
            self.stop = False
            return

        if self.early_dict['mode'] == 'min':
            improve = self.best_matrics - current_metrics
        else:
            improve = current_metrics - self.best_matrics

        if improve > 0:
            self.best_matrics = current_metrics
            self.best_model_para = self.model.state_dict()
            self.patient = 0
        else:
            self.patient += 1

        if self.patient > self.early_dict['patient']:
            self.stop = True
        else:
            self.stop = False

    def train(self, epoch=50, sep=3):
        self.init_history()
        self.eval('train')
        self.eval('val')
        self.info('Begin with: ')
        for i in range(epoch):
            self.optimize()

            self.eval('val')
            if i % sep == 0:
                self.eval('train')
                self.info('Epoch {}: '.format(i))
            if self.early_stopping:
                self.check()
                if self.stop:
                    break
        print('Finish Train')

    def info(self, prefix):
        print(prefix, end='')
        for m in self.history:
            print(m + ' : {:.4f} '.format(self.history[m][-1]), end='')
        print()

    def calcu_metrics(self, metric, output, target):
        return getattr(self, metric)(output, target)

    def acc(self, output, target):
        pred = output.data.max(1, keepdim=True)[1]
        acc = pred.eq(target.data.view_as(pred)).sum()
        return acc

    def eval(self, mode='val', test_dataloader=None, verbose=0):
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
            dataloader = test_dataloader
            if test_dataloader == None:
                print('Please offer test_loader!!')
                return -1
        else:
            raise KeyError

        with torch.no_grad():

            for data, target in tqdm(dataloader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.loss(output, target).item()
                for metrics in self.added_metrics:
                    value[metrics] += self.calcu_metrics(metrics, output, target)

        test_loss /= len(dataloader.dataset)
        if mode != 'test':
            self.history[mode + '_loss'].append(test_loss)
        for metrics in self.added_metrics:
            value[metrics] /= len(dataloader.dataset)
            if mode != 'test':
                self.history[mode + '_' + metrics].append(value[metrics])

        if verbose == 1:
            print(mode + 'set: loss {:.4f} '.format(test_loss, ), end='')
            for metrics in self.added_metrics:
                print(metrics + ' {:.4f}'.format(value[metrics]), end=' ')
            print()

    def pred(self, test_dataloader):
        outputs = []
        with torch.no_grad():
            for data, target in test_dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                outputs.append(output)

        return torch.cat(outputs, dim=-1)

    def save(self, path):
        if self.early_stopping:
            torch.save(self.best_model_para, path)
        else:
            torch.save(self.model.state_dict(), path)

    def plot_history(self, metrics='loss'):
        train_m = 'train_' + metrics
        test_m = 'val_' + metrics

        sns.lineplot(x=range(len(self.history[train_m])), y=self.history[train_m], label=train_m)
        sns.lineplot(x=range(len(self.history[test_m])), y=self.history[test_m], label=test_m)
