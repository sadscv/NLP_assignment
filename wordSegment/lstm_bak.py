#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-4-2 上午1:55
# @Author  : sadscv
# @File    : lstm_bak.py


#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-4-1 下午5:23
# @Author  : sadscv
# @File    : lstm.py
import os
import theano
import theano.tensor as T
import numpy as np

class LSTM(object):

    def __init__(self, embedding_dim, hidden_dim, num_clas, wind_size, vocab_size, bptt_truncate=-1):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_clas = num_clas
        self.wind_size = wind_size
        self.vocab_size = vocab_size
        self.bptt_truncate = bptt_truncate

        self.E = theano.shared(np.random.uniform(-np.sqrt(1./embedding_dim), np.sqrt(1./embedding_dim),
                                                 (vocab_size+1, embedding_dim)).astype(theano.config.floatX))
        self.U = theano.shared(np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim),
                                                 (4, embedding_dim * wind_size, hidden_dim)).astype(theano.config.floatX))
        self.W = theano.shared(np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim),
                                                 (4, embedding_dim, embedding_dim)).astype(theano.config.floatX))
        self.V = theano.shared(np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim),
                                                 (embedding_dim, num_clas)).astype(theano.config.floatX))
        self.b = theano.shared(np.zeros((4, hidden_dim)).astype(theano.config.floatX))
        self.c = theano.shared(np.zeros(num_clas).astype(theano.config.floatX))

        self.params = [self.E, self.U, self.W, self.V, self.b, self.c]
        self.names = ['E', 'U', 'W', 'V', 'b', 'c']

        self.__theano_build__()

    def __theano_build__(self):

        idxs = T.imatrix()
        x = self.E[idxs].reshape((idxs.shape[0], self.wind_size * self.embedding_dim))
        y = T.ivector('y')

        def forward_prop_step(x_t, h_t_prev, c_t_prev):

            input_gate_t = T.nnet.hard_sigmoid(T.dot(x_t, self.U[0]) + T.dot(h_t_prev, self.W[0] + self.b[0]))
            forget_gate_t = T.nnet.hard_sigmoid(T.dot(x_t, self.U[1]) + T.dot(h_t_prev, self.W[1] + self.b[1]))
            output_gate_t = T.nnet.hard_sigmoid(T.dot(x_t, self.U[2]) + T.dot(h_t_prev, self.W[2] + self.b[2]))
            input_knowledge_t = T.tanh(T.dot(x_t, self.U[3]) + T.dot(h_t_prev, self.W[3] + self.b[3]))

            cell_t = c_t_prev * forget_gate_t + input_gate_t * input_knowledge_t
            h_t = T.tanh(cell_t) * output_gate_t

            #返回一个1*n的矩陈， 我们只需要一个向量，所以取[0]
            output_t = T.nnet.softmax(T.dot(h_t, self.V) + self.c)[0]

            return [output_t, h_t, cell_t]

        [o, h, c], _ = theano.scan(
            forward_prop_step,
            sequences= x,
            truncate_gradient=self.bptt_truncate,
            outputs_info = [None,
                            dict(initial=T.zeros(self.hidden_dim)),
                            dict(initial=T.zeros(self.hidden_dim))]
        )

        prediction = T.argmax(o, axis=1)
        #loss
        o_err = T.sum(T.nnet.categorical_crossentropy(o, y))

        self.l2_sqr = 0
        self.l2_sqr += (self.W ** 2).sum()
        self.l2_sqr += (self.U ** 2).sum()
        self.l2_sqr += (self.V ** 2).sum()

        cost = o_err + 0.0002 * self.l2_sqr

        dE = T.grad(cost, self.E)
        dU = T.grad(cost, self.U)
        dW = T.grad(cost, self.W)
        dV = T.grad(cost, self.V)
        db = T.grad(cost, self.b)
        dc = T.grad(cost, self.c)

        self.predict = theano.function([idxs], o)
        self.predict_class = theano.function([idxs], prediction)
        self.ce_err = theano.function([idxs, y], cost)

        lr = T.scalar('learning_rate')

        self.sgd_step = theano.function(
            [idxs, y, lr],
            [],
            updates=[(self.E, self.E - lr * dE),
                       (self.U, self.U - lr * dU),
                       (self.W, self.W - lr * dW),
                       (self.V, self.V - lr * dV),
                       (self.b, self.b - lr * db),
                       (self.c, self.c - lr * dc)]
        )

        def caculate_total_loss(self, X, Y):
            return np.sum([self.ce_err(x, y) for x, y in zip(X, Y) if len(y) >= 5])

        def caculate_loss(self, X, Y):
            num_words = np.sum([len(y) for y in Y])
            return caculate_total_loss(X, Y) / float(num_words)

        def save_model(self, folder):
            for param, name in zip(self.params, self.names):
                np.save(os.path.join(folder, name + '.npy'), param.get_value())
            print("save model to %s" % folder)

