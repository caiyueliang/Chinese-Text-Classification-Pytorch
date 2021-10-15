# coding: UTF-8
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'TextCNN_cyl'

        self.train_path = os.path.join(dataset, 'train.pt')                 # 训练集
        self.dev_path = os.path.join(dataset, 'dev.pt')                     # 验证集
        self.test_path = os.path.join(dataset, 'test.pt')                   # 测试集

        self.class_list = [x.strip() for x in open(
            os.path.join(dataset, 'class.txt'), encoding='utf-8').readlines()]          # 类别名单

        # self.vocab_path = dataset + '/data/vocab.pkl'                                 # 词表

        self.save_dir = os.path.join(dataset, 'save_models')
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_path = os.path.join(self.save_dir, self.model_name + '.ckpt')         # 模型训练结果

        self.log_path = os.path.join(dataset, 'log', self.model_name)

        # self.embedding_pretrained = torch.tensor(
        #     np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32')) \
        #     if embedding != 'random' else None                                        # 预训练词向量

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')      # 设备

        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        # self.n_vocab = 0                                              # 词表大小，在运行时赋值
        self.num_epochs = 20                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率

        # self.embed = self.embedding_pretrained.size(1)\
        #     if self.embedding_pretrained is not None else 300         # 字向量维度
        self.embed = 21

        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)


'''Convolutional Neural Networks for Sentence Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        # if config.embedding_pretrained is not None:
        #     self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        # else:
        #     self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        # print(" =============================== ")
        # x = x[0]                            # x是一个tuple, 这边只用了第一个元素，即文本的索引。[batch_size, seq_len] = [128, 32]
        # print("[TextCNN] x: {} {}".format(x.size(), x.type()))
        # print("[TextCNN] x: {}".format(x))

        out = x
        # out = self.embedding(x)
        # print("[TextCNN] after [embedding]: {}".format(out.size()))

        out = out.unsqueeze(1)              # 多一维: [batch_size, 1, seq_len] = [128, 1, 32]
        # print("[TextCNN] after [unsqueeze]: {}".format(out.size()))
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        # print("[TextCNN] after [conv_and_pool]: {}".format(out.size()))
        out = self.dropout(out)
        # print("[TextCNN] after [dropout]: {}".format(out.size()))
        out = self.fc(out)
        # print("[TextCNN] after [fc]: {}".format(out.size()))

        return out
