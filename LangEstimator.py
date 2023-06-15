import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import seaborn as sns
from torchsummary import summary
import numpy as np
from MyEstimator import Estimator
from torch.autograd import Variable

def ls_discriminator_loss(scores_real, scores_fake, device):
    """
    Compute the Least-Squares GAN loss for the discriminator.

    Inputs:
    - scores_real: PyTorch Variable of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Variable of shape (N,) giving scores for the fake data.

    Outputs:
    - loss: A PyTorch Variable containing the loss.
    """
    N = scores_real.size()
    #     print(N)

    true_labels = Variable(torch.ones(N)).to(device)

    fake_image_loss = (torch.mean((scores_real - true_labels) ** 2))
    real_image_loss = (torch.mean((scores_fake) ** 2))

    loss = 0.5 * fake_image_loss + 0.5 * real_image_loss

    return loss


def ls_generator_loss(scores_fake, device):
    """
    Computes the Least-Squares GAN loss for the generator.

    Inputs:
    - scores_fake: PyTorch Variable of shape (N,) giving scores for the fake data.

    Outputs:
    - loss: A PyTorch Variable containing the loss.
    """
    N = scores_fake.size()

    true_labels = Variable(torch.ones(N)).to(device)

    loss = 0.5 * ((torch.mean((scores_fake - true_labels) ** 2)))

    return loss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)





class movieEstimator(Estimator):

    def preprocess_train(self):
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=2, gamma=0.1)


    def postprocess_epoch(self):
        self.scheduler.step()

    def calcu_loss(self, item):
        data, target, batch_seq_len = item[0], item[1], item[2]
        data, target = data.to(self.device), target.to(self.device)

        output, hidden = self.model(data, batch_seq_len)
        return self.loss(output, target), output, target

    def smy(self, model, input_size):
        print(f'The model has {count_parameters(model):,} trainable parameters.')


class tangEstimator(Estimator):
    def __init__(self, model: torch.nn.Module, input_size, device, metrics=None):
        super(tangEstimator, self).__init__(model, input_size, device, metrics)
        self.discriminator = None

    def register_gan(self, model):
        self.discriminator = model
        self.discriminator.to(self.device)

        self.history['train_G_loss'] = []
        self.history['train_D_loss'] = []


    def preprocess_train(self):
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=10, gamma=0.1)


    def postprocess_epoch(self):
        self.scheduler.step()

    def smy(self, model, input_size):
        print(f'The model has {count_parameters(model):,} trainable parameters.')

    def gan_loss(self, data):
        data = data.to(self.device)
        generated_sample = self.model.generate(data[:, :6])
        scores_fake = self.discriminator(generated_sample)
        scores_real = self.discriminator(data)
        D_loss = ls_discriminator_loss(scores_real, scores_fake, self.device)
        G_loss = ls_generator_loss(scores_fake, self.device)
        return D_loss, G_loss


    def calcu_loss(self, item):
        data, target = item[0], item[1]
        batch_size, seq_len = data.size()
        data, target = data.to(self.device), target.to(self.device)
        output, hidden = self.model(data)
        output = output.reshape(batch_size * seq_len, -1)
        target = target.view(-1)

        pred_loss = self.loss(output, target)
        return pred_loss, output, target

    def optimize_gan(self, i):
        for k, v in self.model.named_parameters():
            if 'embeddings' in k:
                v.requires_grad = False

        lr = self.scheduler.get_last_lr()[-1]
        self.D_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.discriminator.parameters()), lr=lr, betas=(0.5, 0.999))
        self.G_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, betas=(0.5, 0.999))

        self.model.train()
        loop = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader))
        D_train_loss, G_train_loss =  0., 0.
        num = 0
        value = {}
        for metrics in self.added_metrics:
            value[metrics] = 0.

        for batch_idx, (data, target) in loop:
            loop.set_description('GAN Training Phase Epoch %i' % i)
            self.optim.zero_grad()

            D_loss, G_loss = self.gan_loss(data)
            G_loss.backward(retain_graph=True)
            self.optim.step()

            self.D_optim.zero_grad()
            D_loss.backward()
            self.D_optim.step()



            with torch.no_grad():
                D_train_loss += D_loss.item()
                G_train_loss += G_loss.item()
                num += data.shape[0]
                postfix = {}

                postfix['Generator_loss'] = ' {:.6f}'.format(G_train_loss / (batch_idx + 1))
                postfix['Discriminator_loss'] = ' {:.6f}'.format(D_train_loss / (batch_idx + 1))
                loop.set_postfix(postfix)

        self.history['train_G_loss'].append(G_train_loss / len(self.train_dataloader))  # 记录训练过程的loss
        self.history['train_D_loss'].append(D_train_loss / len(self.train_dataloader))
        for k, v in self.model.named_parameters():
            if 'embeddings' in k:
                v.requires_grad = True

    def train(self, epoch=50):
        super(tangEstimator, self).train(epoch)
        if self.discriminator == None:
            return

        self.optimize_gan(epoch)



    def generate(self, start_words, ix2word, word2ix, max_gen_len=48):
        results = list(start_words)
        start_words_len = len(start_words)
        # 第一个词语是<START>
        input = torch.Tensor([word2ix['<START>']]).view(1, 1).long()

        hidden = None
        self.model.eval()
        with torch.no_grad():
            for i in range(max_gen_len):
                input = input.to(self.device)

                output, hidden = self.model(input, hidden)
                # 如果在给定的句首中，input 为句首中的下一个字
                if i < start_words_len:
                    w = results[i]
                    input = input.data.new([word2ix[w]]).view(1, 1)
                # 否则将 output 作为下一个 input 进行
                else:
                    top_index = output.data[0].topk(1)[1][0].item()
                    w = ix2word[top_index]
                    results.append(w)
                    input = input.data.new([top_index]).view(1, 1)
                if w == '<EOP>':
                    del results[-1]
                    break
        poem = ''.join(results)
        return poem