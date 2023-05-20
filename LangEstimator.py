import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import seaborn as sns
from torchsummary import summary
import numpy as np
from MyEstimator import Estimator


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class langEstimator(Estimator):
    def __init__(self, model: torch.nn.Module, input_size, device, metrics=None):
        super(langEstimator, self).__init__(model, input_size, device, metrics)


    def smy(self, model, input_size):
        print(f'The model has {count_parameters(model):,} trainable parameters.')

    def calcu_loss(self, data, target):

        data, target = data.to(self.device), target.to(self.device)
        target = target.view(-1)
        output, hidden = self.model(data)
        #print(output.shape, target.shape)
        return self.loss(output, target), output, target


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
                hidden = hidden.to(self.device)
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

        return results