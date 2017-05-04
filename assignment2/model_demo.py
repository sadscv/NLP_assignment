#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-3-28 下午2:10
# @Author  : sadscv
# @File    : model_demo.py

#config class
class Config(object):
    pass


#model class
class Model(object):
    #init func
    def __init__(self):
        pass

    def run_epoch(self, sess, input_data, input_labels):
        pass

    def fit(self, sess, input_data, input_labels):
        pass

    def predict(self, sess, input_data, input_labels=None):
        pass


    def add_model(self, input_data):
        pass

    #load data func
    def load_data(self):
        pass

    #add placeholder func
    def add_placeholder(self):
        pass

    def add_loss_op(self, pred):
        pass

    #create feeddict func
    def create_feed_dict(self, input_batch, label_batch):
        pass

    #train operation
    def add_training_op(self):
        pass



    pass


