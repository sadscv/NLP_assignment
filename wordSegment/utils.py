#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-4-1 下午5:20
# @Author  : sadscv
# @File    : utils.py
import re
import os
import numpy as np

def load_vocabulary(path='./runs/vocab'):
    """
    载入 词汇-编号 字典
    :param path:path of the vocab file
    :return: dict  {vocab:num}
    """
    vocabulary = {}
    with open(path, 'r') as f:
        for line in f:
            split = line.strip('\n').split(' ')
            vocabulary[split[0]] = int(split[1])
    return vocabulary


def context_win(l, window_size):
    """
    滑动窗口大小为windows_size,在原L的两端分别加上windows_size//2的[-1],在新的L上滑动，每个context大小为window_size,共有len(l)个。
    :param l: 输入list e.g. [32, 12, 17, 0, 23, 21],代表一个句子,用数字表示其中每个字
    :param window_size:
    :return:
    """
    assert(window_size % 2) == 1
    assert window_size > 1
    l = list(l)

    lpadded = window_size // 2 * [-1] + l + window_size // 2 * [-1]
    out = [lpadded[i:(i+window_size)] for i in range(len(l))]

    assert len(out) ==  len(l)
    return np.array(out, dtype=np.int32)


def load_data(path, window_size=7):

    X_train, Y_train = [], []
    vocabulary = load_vocabulary()
    with open(path) as f:
        for line in f:
            split = re.split('\s+', line.strip())
            y = []
            for word in split:
                length = len(word)
                if length == 1:
                    y.append(3)
                else:
                    y.extend([0] + [1] * (length - 2) + [2])
            newline = ''.join(split)#?
            x = [vocabulary[char] if vocabulary.get(char) else 0 for char in newline]
            X_train.append(context_win(x, window_size))
            Y_train.append(y)

    return X_train, Y_train, vocabulary

def train_with_sgd(model, X_train, y_train, learning_rate=0.001, nepoch=20, callback_every=10000, callback=None):
    num_example_seen = 0
    for epoch in range(nepoch):
        for i in np.random.permutation(len(y_train)):
            if len(y_train[i]) < 3:
                continue
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_example_seen +=1
            if (callback and callback_every and num_example_seen % callback_every == 0 ):
                callback(model, num_example_seen)


def convert_predicts_to_segments(predicts, seq):
    assert len(predicts) == len(seq)
    i = 0
    segs = []
    while(i < len(seq)):
        if predicts[i] == 0:
            j = i + 1
            while(j < len(seq) and predicts[j] != 2):
                j += 1
            if j == len(seq):
                segs.append(seq[i:j])
            else:
                segs.append(seq[i:j+1])
            i = j + 1
        if i < len(seq) and predicts[i] != 0:
            segs.append(seq[i])
            i += 1
    return segs


def load_model(folder, modelClass, hyperparams):
    print("loading model from %s" % folder)
    print("...")

    E = np.load(os.path.join(folder, 'E.npy'))
    U = np.load(os.path.join(folder, 'U.npy'))
    W = np.load(os.path.join(folder, 'W.npy'))
    V = np.load(os.path.join(folder, 'V.npy'))
    b = np.load(os.path.join(folder, 'b.npy'))
    c = np.load(os.path.join(folder, 'c.npy'))

    hidden_dim = hyperparams['hidden_dim']
    embedding_dim = hyperparams['embedding_dim']
    vocab_size = hyperparams['vocab_size']
    num_clas = hyperparams['num_clas']
    wind_size = hyperparams['wind_size']

    model = modelClass(embedding_dim, hidden_dim, num_clas, wind_size, vocab_size)
    model.E.set_value(E)
    model.U.set_value(U)
    model.W.set_value(W)
    model.V.set_value(V)
    model.b.set_value(b)
    model.c.set_value(c)
    print("lstm model has been loaded")
    return model