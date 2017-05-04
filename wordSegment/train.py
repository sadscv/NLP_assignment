#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-4-1 下午5:17
# @Author  : sadscv
# @File    : train.py
import os

import sys

from utils import load_data, train_with_sgd
from lstm import LSTM

class Config:
    def __init__(self):
        pass


# Model Hyperparams
LEARNING_RATE =  float(os.environ.get("LEARNING_RATE", "0.001"))
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "50"))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", "128"))
NUM_CLASSES = int(os.environ.get("NUM_CLASSES", "4"))
WIN_SIZE = int(os.environ.get("WIN_SIZE", 7))

# Training parameters
NEPOCH = int(os.environ.get("NEPOCH", "20"))
PRINT_EVERY = int(os.environ.get("PRINT_EVERY", "2000"))

#load data
X_train, y_train, vocab = load_data('./data/msr_training.utf8')
X_test, y_test, _ = load_data('./data/msr_test_gold.utf8')

#build model
model = LSTM(EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES, WIN_SIZE, len(vocab), bptt_truncate=-1)

def sgd_callback(model, num_example_seen):
    loss = model.calculate_loss(X_test, y_test)
    print("\n num_example_seen: %d" % (num_example_seen))
    print("_______________________________________")
    print("Loss:%f", loss)
    folder = os.path.abspath(os.path.join(os.path.curdir, "runs"))
    model.save_model(folder)
    sys.stdout.flush()

print('start epoch')
for epoch in range(NEPOCH):
    train_with_sgd(model, X_train, y_train, LEARNING_RATE, nepoch=1, callback_every=PRINT_EVERY, callback=sgd_callback)

