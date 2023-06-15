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
tang_input_size = (48,)
tang_vocab_size = 8293
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

mnist_root = './data/'

kaggle_test_path = './data/KAGGLE/toy_test/'
kaggle_pred_path = './data/KAGGLE/test/'
kaggle_train_path = './data/KAGGLE/toy_train/'

tang_train_path = './data/TANG/tang.npz'

movie_dataset_path = './data/MOVIE/Dataset/'
movie_train_path, movie_val_path = movie_dataset_path + 'train.txt', movie_dataset_path + 'validation.txt'
movie_test_path = movie_dataset_path + 'test.txt'
pred_word2vec_path = movie_dataset_path  + 'wiki_word2vec_50.bin'

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


import numpy as np


class PoemDataSet(torch.utils.data.Dataset):
    def __init__(self,poem_path,seq_len):
        self.seq_len = seq_len
        self.poem_path = poem_path
        self.poem_data, self.ix2word, self.word2ix = get_tang_raw_data(poem_path)
        self.no_space_data = self.filter_space()

    def __getitem__(self, idx:int):
        txt = self.no_space_data[idx*self.seq_len : (idx+1)*self.seq_len]
        label = self.no_space_data[idx*self.seq_len + 1 : (idx+1)*self.seq_len + 1] # 将窗口向后移动一个字符就是标签

        txt = torch.from_numpy(np.array(txt)).long()
        label = torch.from_numpy(np.array(label)).long()
        return txt,label

    def __len__(self):
        return int(len(self.no_space_data) / self.seq_len)

    def filter_space(self): # 将空格的数据给过滤掉，并将原始数据平整到一维
        t_data = torch.from_numpy(self.poem_data).view(-1)
        flat_data = t_data.numpy()
        no_space_data = []
        for i in flat_data:
            if (i != 8292 ):
                no_space_data.append(i)
        return no_space_data




def get_tang_train_dataloader(batch_size=16, rate=0.8):
    dataset_train = PoemDataSet(tang_train_path, tang_input_size[0])
    dataset_train, dataset_val = dataset_split(dataset_train, rate)
    train_dataloader = torch.utils.data.DataLoader(dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2)
    val_dataloader = torch.utils.data.DataLoader(dataset_val,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=2)

    return train_dataloader, val_dataloader



def get_tang_raw_data(poem_path):
    datas = np.load(poem_path,allow_pickle=True)  #numpy 1.16.2  以上引入了allow_pickle
    #datas = np.load(self.poem_path)
    data = datas['data']
    ix2word = datas['ix2word'].item()
    word2ix = datas['word2ix'].item()
    return data, ix2word, word2ix




# 简繁转换 并构建词汇表
def build_word_dict(train_path):
    words = []
    max_len = 0
    total_len = 0
    with open(train_path,'r',encoding='UTF-8') as f:
        lines = f.readlines()
        for line in  lines:
            line = convert(line, 'zh-cn') #转换成大陆简体
            line_words = re.split(r'[\s]', line)[1:-1] # 按照空字符\t\n 空格来切分
            max_len = max(max_len, len(line_words))
            total_len += len(line_words)
            for w in line_words:
                words.append(w)
    words = list(set(words))#最终去重
    words = sorted(words) # 一定要排序不然每次读取后生成此表都不一致，主要是set后顺序不同
    #用unknown来表示不在训练语料中的词汇
    word2ix = {w:i+1 for i,w in enumerate(words)} # 第0是unknown的 所以i+1
    ix2word = {i+1:w for i,w in enumerate(words)}
    word2ix['<unk>'] = 0
    ix2word[0] = '<unk>'
    avg_len = total_len / len(lines)
    return word2ix, ix2word, max_len,  avg_len


import gensim # word2vec预训练加载
import jieba #分词
from zhconv import convert #简繁转换
# 变长序列的处理
from torch.nn.utils.rnn import pad_sequence
import re

def build_word2id(file):
    """
    :param file: word2id #保存地址
    :return: None
    """
    word2id = {'_PAD_': 0}
    path = []
    print(path)
    for _path in path:
        with open(_path, encoding='utf-8') as f:
            for line in f.readlines():
                sp = line.strip().split()
                for word in sp[1:]:
                    if word not in word2id.keys():
                        word2id[word] = len(word2id)
    with open(file, 'w', encoding='utf-8') as f:
        for w in word2id:
            f.write(w+'\t')
            f.write(str(word2id[w]))
            f.write('\n')


def mycollate_fn(data):
    # 这里的data是getittem返回的（input，label）的二元组，总共有batch_size个
    data.sort(key=lambda x: len(x[0]), reverse=True)  # 根据每个句子的长度进行排序，长的排前
    data_length = [len(sq[0]) for sq in data] # 记录句子的长度，在mini-batch中方便压缩
    input_data = []
    label_data = []
    for i in data:
        input_data.append(i[0])
        label_data.append(i[1])
    input_data = pad_sequence(input_data, batch_first=True, padding_value=0)
    label_data = torch.tensor(label_data)
    return input_data, label_data, data_length


class CommentDataSet(torch.utils.data.Dataset):
    def __init__(self, data_path, word2ix, ix2word):
        self.data_path = data_path
        self.word2ix = word2ix
        self.ix2word = ix2word
        self.data, self.label = self.get_data_label()

    def __getitem__(self, idx: int):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)

    def get_data_label(self):
        data = []
        label = []
        with open(self.data_path, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
            for line in lines:
                try:
                    label.append(torch.tensor(int(line[0]), dtype=torch.int64))
                except BaseException:  # 遇到首个字符不是标签的就跳过比如空行，并打印
                   # print('not expected line:' + line)
                    continue
                line = convert(line, 'zh-cn')  # 转换成大陆简体
                line_words = re.split(r'[\s]', line)[1:-1]  # 按照空字符\t\n 空格来切分
                words_to_idx = []
                for w in line_words:
                    try:
                        index = self.word2ix[w]
                    except BaseException:
                        index = 0  # 测试集，验证集中可能出现没有收录的词语，置为0
                    words_to_idx.append(index)
                data.append(torch.tensor(words_to_idx, dtype=torch.int64))
        return data, label

def get_movie_dataloader(batch_size=32):
    word2ix, ix2word, max_len, avg_len = build_word_dict(movie_train_path)
    train_data = CommentDataSet(movie_train_path, word2ix, ix2word)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=0, collate_fn=mycollate_fn, )

    validation_data = CommentDataSet(movie_val_path, word2ix, ix2word)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=True,
                                   num_workers=0, collate_fn=mycollate_fn, )

    test_data = CommentDataSet(movie_test_path, word2ix, ix2word)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                             num_workers=0, collate_fn=mycollate_fn, )

    return train_loader, validation_loader, test_loader

