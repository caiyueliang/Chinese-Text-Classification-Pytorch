# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
import random
from datetime import timedelta


MAX_VOCAB_SIZE = 10000
UNK, PAD = '<UNK>', '<PAD>'


def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[0]
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


# =========================================================================
def random_mask_rows(tensor_list: list, num_mask=1, max_mask=1):
    new_tensor_list = []

    for tensor in tensor_list:
        # print("==========================", flush=True)
        # print(tensor.size(), flush=True)
        # print(tensor, flush=True)
        max_len = tensor.size()[0]

        # time mask
        for i in range(num_mask):
            start = random.randint(0, max_len - 1)
            length = random.randint(1, max_mask)
            end = min(max_len, start + length)

            # print("max_len:{}, start:{}, end:{}".format(max_len, start, end), flush=True)
            tensor[start:end, :] = 0

        # print(tensor, flush=True)
        # print("==========================", flush=True)
        new_tensor_list.append(tensor)

    return new_tensor_list


def random_mask_columns(tensor_list: list, num_mask=1, max_mask=1):
    new_tensor_list = []

    for tensor in tensor_list:
        # print("==========================", flush=True)
        # print(tensor.size(), flush=True)
        # print(tensor, flush=True)
        max_len = tensor.size()[1]

        # freq mask
        for i in range(num_mask):
            start = random.randint(0, max_len - 1)
            length = random.randint(1, max_mask)
            end = min(max_len, start + length)

            # print("max_len:{}, start:{}, end:{}".format(max_len, start, end), flush=True)
            tensor[:, start:end] = 0

        # print(tensor, flush=True)
        # print("==========================", flush=True)
        new_tensor_list.append(tensor)

    return new_tensor_list


# =========================================================================
def build_dataset(config, ues_word):
    """
    返回的数据的格式：[([...], 0), ([...], 1), ...]
    我这里的[...]，应该是一个矩阵
    """
    def load_dataset(path, pad_size=32):
        contents = torch.load(f=path)
        return contents

    train = load_dataset(config.train_path, config.pad_size)
    print("[build_dataset] train len: {}".format(len(train)), flush=True)
    dev = load_dataset(config.dev_path, config.pad_size)
    print("[build_dataset] dev len: {}".format(len(dev)), flush=True)
    test = load_dataset(config.test_path, config.pad_size)
    print("[build_dataset] test len: {}".format(len(test)), flush=True)

    return 0, train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device, train_flag=False):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size

        self.residue = False  # 记录batch数量是否为整数 
        if len(batches) % self.n_batches != 0:
            self.residue = True

        self.index = 0
        self.device = device
        self.train_flag = train_flag

    def _to_tensor(self, data):
        if self.train_flag is False:
            # dev set or test set
            tensor_list = [_[0] for _ in data]
            x = torch.stack(tensor_list, dim=0).to(self.device)
            x = x.to(torch.float32)
        else:
            # train set, use Data Augmentation
            tensor_list = [_[0] for _ in data]
            tensor_list = random_mask_rows(tensor_list, num_mask=2, max_mask=4)         # 行 mask
            tensor_list = random_mask_columns(tensor_list, num_mask=2, max_mask=4)      # 列 mask
            x = torch.stack(tensor_list, dim=0).to(self.device)
            x = x.to(torch.float32)

        y = torch.LongTensor([_[1] for _ in data]).to(self.device)

        return x, y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config, train_flag=False):
    iter = DatasetIterater(dataset, config.batch_size, config.device, train_flag=train_flag)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


if __name__ == "__main__":
    data = list()

    # a = [1, 2, 3]
    # b = [2, 3, 4]
    # a = [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]
    # b = [[2, 3, 4], [2, 3, 4], [2, 3, 4], [2, 3, 4]]

    a = torch.tensor([[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4]])
    b = torch.tensor([[5, 5, 5, 5, 5], [6, 6, 6, 6, 6], [7, 7, 7, 7, 7], [8, 8, 8, 8, 8]])

    data.append(a)
    data.append(b)

    print(data)

    # x = torch.Tensor(data)
    # x = torch.cat(data, dim=0)
    # my_x = torch.stack(data, dim=0)
    # print(my_x)
    # print(my_x.size())

    # '''提取预训练词向量'''
    # vocab_dir = "./THUCNews/data/vocab.pkl"
    # pretrain_dir = "./THUCNews/data/sgns.sogou.char"
    # emb_dim = 300
    # filename_trimmed_dir = "./THUCNews/data/vocab.embedding.sougou"
    # word_to_id = pkl.load(open(vocab_dir, 'rb'))
    # embeddings = np.random.rand(len(word_to_id), emb_dim)
    # f = open(pretrain_dir, "r", encoding='UTF-8')
    # for i, line in enumerate(f.readlines()):
    #     # if i == 0:  # 若第一行是标题，则跳过
    #     #     continue
    #     lin = line.strip().split(" ")
    #     if lin[0] in word_to_id:
    #         idx = word_to_id[lin[0]]
    #         emb = [float(x) for x in lin[1:301]]
    #         embeddings[idx] = np.asarray(emb, dtype='float32')
    # f.close()
    # np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)

    data = random_mask_rows(data, num_mask=1, max_mask=1)
    data = random_mask_columns(data, num_mask=1, max_mask=1)
    print(data)
    data = torch.stack(data, dim=0)
    print(data)
    print(data.size())
