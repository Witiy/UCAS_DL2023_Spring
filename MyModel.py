import torch.nn as nn
import torch

class MnistCNN(nn.Module):
    def __init__(self):
        super(MnistCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            #nn.BatchNorm2d(16)
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10)
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.out(x)
        return x


class KaggleCNN(nn.Module):
    def __init__ (self):
        super(KaggleCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)

        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 7),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)

        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 9),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)

        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        #input (bs, 3, 150, 150)
        x = self.conv1(x) #input (bs, 32, 111, 111)
        x = self.conv2(x) #input (bs, 64, 53, 53)
        x = self.conv3(x)#input (bs, 64, 23, 23)
        x = self.conv4(x)#input (bs, 64, 7, 7)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
# import torch.nn.functional as F

class TangModel_LSTM(nn.Module):
    def __init__(self, vocab_size):
        super(TangModel_LSTM, self).__init__()
        self.hidden_dim = 512
        self.embedding_dim = 256
        self.num_layers = 3

        self.embeddings = nn.Embedding(vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True, dropout=0.1)

        self.linear1 = nn.Linear(self.hidden_dim, 4096)


        self.linear2 = nn.Linear(4096, vocab_size)


    def forward(self, input, hidden=None):
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        embeds = self.embeddings(input.long())
        # [batch, seq_len] => [batch, seq_len, embed_dim]

        if hidden is None:
            h_0 = input.data.new(self.num_layers, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(self.num_layers, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden
        output, hidden = self.lstm(embeds, (h_0, c_0))

        output = torch.tanh(self.linear1(output))

        output = self.linear2(output)

        return output, hidden

    def generate(self, input, hidden=None, max_gen_len=48 - 6):
        pred = input
        for i in range(max_gen_len):
            pred, hidden = self.forward(pred, hidden)
            pred = pred.data.topk(1)[1][:, -1, :]
            input = torch.concat([input, pred], dim=1)

        return input

class TangModel_Discriminator(nn.Module):
    def __init__(self, embeddings, embedding_dim):
        super(TangModel_Discriminator, self).__init__()
        self.hidden_dim = 512
        self.embedding_dim = embedding_dim
        self.num_layers = 3
        self.embeddings = embeddings
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True,
                            dropout=0)

        self.linear = nn.Linear(self.hidden_dim , 1)


    def forward(self, input, hidden=None):
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        embeds = self.embeddings(input.long())

        # [batch, seq_len] => [batch, seq_len, embed_dim]

        if hidden is None:
            h_0 = input.data.new(self.num_layers, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(self.num_layers, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden
        output, (h, c) = self.lstm(embeds, (h_0, c_0))
        h = h[-1]
        output = self.linear(h)
        return output


from MyDataset import pred_word2vec_path, build_word_dict, movie_train_path
import gensim

def get_pre_weight():
    word2ix, ix2word, max_len, avg_len = build_word_dict(movie_train_path)
    vocab_size = len(word2ix)
    pre_model = gensim.models.KeyedVectors.load_word2vec_format(pred_word2vec_path, binary=True)
    weight = torch.zeros(vocab_size, 50)
    #初始权重
    for i in range(len(pre_model.index_to_key)):#预训练中没有word2ix，所以只能用索引来遍历
        try:
            index = word2ix[pre_model.index_to_key[i]]#得到预训练中的词汇的新索引
        except:
            continue
        weight[index, :] = torch.from_numpy(pre_model.get_vector(
            ix2word[word2ix[pre_model.index_to_key[i]]]))#得到对应的词向量
    return weight



from torch.nn.utils.rnn import pack_padded_sequence

class MovieModel(nn.Module):
    def __init__(self):
        super(MovieModel, self).__init__()
        self.embedding_dim = 50
        self.hidden_dim = 128
        self.layer_num = 2

        embedding_weight = get_pre_weight()
        self.embeddings = nn.Embedding.from_pretrained(embedding_weight)
        # requires_grad指定是否在训练过程中对词向量的权重进行微调
        self.embeddings.weight.requires_grad = True
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=self.layer_num,
                            batch_first=True, dropout=0.1,)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(self.hidden_dim, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 2)


    def forward(self, input, batch_seq_len, hidden=None):
        embeds = self.embeddings(input)  # [batch, seq_len] => [batch, seq_len, embed_dim]
        embeds = pack_padded_sequence(embeds, batch_seq_len, batch_first=True)
        batch_size, seq_len = input.size()
        if hidden is None:
            h_0 = input.data.new(self.layer_num * 1, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(self.layer_num * 1, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden
        output, (h, c) = self.lstm(embeds, (h_0, c_0))
        #用最后一层的隐状态作为判别特征
        output = h[-1]
        output = self.dropout(torch.tanh(self.fc1(output)))
        output = torch.tanh(self.fc2(output))
        output = self.fc3(output)

        return output, hidden





from torchvision import models



class NetFactory:
    def getNet(self, mode='mnist', **kwargs):
        if mode == 'mnist':
            return MnistCNN()
        elif mode == 'kaggle':
            return KaggleCNN()
        elif mode == 'tang':
            return TangModel_LSTM(kwargs['vocab_size'])
        elif mode == 'tang_d':
            return TangModel_Discriminator(kwargs['embeddings'], kwargs['embedding_dim'])
        elif mode == 'movie':
            return MovieModel()
        elif mode == 'resnet18':
            resnet18 = models.resnet18(pretrained=True)
            resnet18.fc = nn.Linear(512, 2)  # 将最后一层全连接的输出调整到2维
            return resnet18
        elif mode == 'resnet50':
            resnet50 = models.resnet50(pretrained=True)
            resnet50.fc = nn.Linear(2048, 2)  # 将最后一层全连接的输出调整到2维
            return resnet50
        else:
            raise KeyError("No exist model!")




