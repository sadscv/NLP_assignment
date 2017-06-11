#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-3-28 下午4:28
# @Author  : sadscv
# @File    : RNNLM_demo.py

import sys
import time
import math

import numpy as np
from copy import deepcopy

from utils import calculate_perplexity, get_ptb_dataset, Vocab
from utils import ptb_iterator, sample

import tensorflow as tf
import LightRNN.embedding_ops_modified as test

from tensorflow.contrib import rnn
# from tensorflow.contrib.legacy_seq2seq import sequence_loss
from loss import sequence_loss
# from tensorflow.python.ops.seq2seq import sequence_loss
# from tf.contrib.seq2seq import sequence_loss
from model import LanguageModel

# Let's set the parameters of our model
# http://arxiv.org/pdf/1409.2329v4.pdf shows parameters that would achieve near
# SotA numbers

class Config(object):
  """Holds model hyperparams and data information.

  The config class is used to store various hyperparameters and dataset
  information parameters. Model objects are passed a Config() object at
  instantiation.
  """
  batch_size = 64
  embed_size = 500
  hidden_size = 100
  num_steps = 10
  max_epochs = 16
  early_stopping = 2
  dropout = 0.9
  lr = 0.001
  lstm_size = 100


class RNNLM_Model(LanguageModel):

    def load_data(self, debug=False):
        """Loads starter word-vectors and train/dev/test data."""
        self.vocab = Vocab()
        self.vocab.construct(get_ptb_dataset('train'))
        self.encoded_train = np.array(
            [self.vocab.encode(word) for word in get_ptb_dataset('train')],
            dtype=np.int32)
        self.encoded_valid = np.array(
            [self.vocab.encode(word) for word in get_ptb_dataset('valid')],
            dtype=np.int32)
        self.encoded_test = np.array(
            [self.vocab.encode(word) for word in get_ptb_dataset('test')],
            dtype=np.int32)
        # matrix_length, 二维词表长度。
        self.matrix_length = math.ceil(math.sqrt(len(self.vocab)))
        if debug:
            num_debug = 1024
            self.encoded_train = self.encoded_train[:num_debug]
            self.encoded_valid = self.encoded_valid[:num_debug]
            self.encoded_test = self.encoded_test[:num_debug]

    def add_placeholders(self):
        """Generate placeholder variables to represent the input tensors
            These placeholders are used as inputs by the rest of the model building
            code and will be fed data during training.  Note that when "None" is in a
            placeholder's shape, it's flexible
            """

        self.input_placeholder = tf.placeholder(dtype=tf.int32,
                                                shape=(None,
                                                       self.config.num_steps),
                                                name='input')
        self.labels_placeholder = tf.placeholder(dtype=tf.int32,
                                                 shape=(None,
                                                        self.config.num_steps),
                                                 name='label')
        self.dropout_placeholder = tf.placeholder(dtype=tf.float32,
                                                  name='dropout')

    def add_embedding(self, name='embed'):
        """Add embedding layer."""
        with tf.device('/gpu:0'):
            with tf.name_scope(name):
# _______________________base_model_without_devided______________________________________#

                # embedding = tf.get_variable('Embedding', [len(self.vocab),
                #                                           self.config.embed_size],
                #                             # trainable=True,
                #                             )
                # inputs = test.embedding_lookup(embedding, self.input_placeholder)
                # inputs = [tf.squeeze(x, [1])
                #           for x in tf.split(inputs, self.config.num_steps, 1)]
                # return inputs
#___________________________________END___________________________________________#


#_______________divided row and column, combine them and integred___________________#

                # embedding_r = tf.get_variable('Eb_r', [self.matrix_length,
                #                                               self.config.embed_size])
                # embedding_c = tf.get_variable('Eb_c', [self.matrix_length,
                #                                               self.config.embed_size])
                # inputs_r = tf.nn.embedding_lookup(embedding_r,
                #                                   self.input_placeholder // self.matrix_length, name='inputs_r')
                # inputs_c = tf.nn.embedding_lookup(embedding_c,
                #                                   self.input_placeholder % self.matrix_length, name='inputs_c')
                # inputs = tf.add(inputs_r, inputs_c, name='inputs')
                # inputs = [tf.squeeze(x, [1])
                #           for x in tf.split(inputs, self.config.num_steps, 1)]
                # return inputs
#____________________________END____________________________________#


#___________divided row and column, apply the next word's row______________________________________#

                embedding_r = tf.get_variable('Eb_r', [self.matrix_length,
                                                              self.config.embed_size])
                embedding_c = tf.get_variable('Eb_c', [self.matrix_length,
                                                              self.config.embed_size])

                self.input_row_indice = self.input_placeholder // self.matrix_length
                self.input_column_indice = self.input_placeholder % self.matrix_length

                inputs_r = tf.nn.embedding_lookup(embedding_r,
                                                  self.input_row_indice, name='inputs_r')
                inputs_c = tf.nn.embedding_lookup(embedding_c,
                                                  self.input_column_indice, name='inputs_c')
                inputs_r_next = tf.nn.embedding_lookup(embedding_r,
                                                       self.labels_placeholder // self.matrix_length, name='inputs_r_next')
                inputs = tf.concat([inputs_r, inputs_c, inputs_r_next], 2, name='combine_r_and_c')

                inputs = [tf.squeeze(x, [1])
                          for x in tf.split(inputs, self.config.num_steps, 1)]
                return inputs
#____________________________END____________________________________#

                # inputs = tf.nn.embedding_lookup(embedding, self.input_placeholder)
                # input_placeholder是2d的placeholder,[batch_size, num_steps]
                # inputs 是个3d的tensor, [batch_size, num_steps,  embed_size]
                # tf.split之后依然是3d tensor, [batch_size, 1, embed_size]
                # tf.squeeze之后变为2d tensor, [batch_size, embed_size]
                # return 的是一个 list, 其中每个元素为上一行介绍的2d tensor.

    def add_model_RNN(self, inputs):
        self.initial_state = tf.zeros([self.config.batch_size,
                                       self.config.hidden_size])
        lstm_cell = rnn.BasicRNNCell(self.config.lstm_size)
        cell = rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.config.dropout)
        self._initial_state = cell.zero_state(batch_size=self.config.batch_size, dtype=tf.float32)
        state = self._initial_state
        rnn_outputs = []
        with tf.variable_scope('RNN') as scope:
            for tstep, current_input in enumerate(inputs):
                if tstep > 0:
                    scope.reuse_variables()
                output, state = cell(current_input, state)
                rnn_outputs.append(state)
            self.final_state = rnn_outputs[-1]

        return rnn_outputs

    def add_model_LSTM(self, inputs):
        self.initial_state = tf.zeros([self.config.batch_size,
                                       self.config.hidden_size])
        lstm_cell = rnn.BasicLSTMCell(self.config.lstm_size)
        cell = rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.config.dropout)
        self._initial_state = cell.zero_state(batch_size=self.config.batch_size, dtype=tf.float32)
        state = self._initial_state
        rnn_outputs = []
        with tf.variable_scope('RNN') as scope:
            for tstep, current_input in enumerate(inputs):
                if tstep > 0:
                    scope.reuse_variables()
                output, state = cell(current_input, state)
                rnn_outputs.append(state)
            self.final_state = rnn_outputs[-1]

        return rnn_outputs


    def add_model(self, inputs):
        """Creates the RNN LM model.

        In the space provided below, you need to implement the equations for the
        RNNLM model. Note that you may NOT use built in rnn_cell functions from
        tensorflow.

              H: (hidden_size, hidden_size)
              I: (embed_size, hidden_size)
              b_1: (hidden_size,)

        Args:
          inputs: List of length num_steps, each of whose elements should be
                  a tensor of shape (batch_size, embed_size).
        Returns:
          outputs: List of length num_steps, each of whose elements should be
                   a tensor of shape (batch_size, hidden_size)
        """

#__________________________________________baseline______________________________________________________#
        # with tf.variable_scope('InputDropout'):
        #     inputs = [tf.nn.dropout(x, self.dropout_placeholder) for x in
        #               inputs]
        #
        # with tf.variable_scope('RNN') as scope:
        #     self.initial_state = tf.zeros([self.config.batch_size,
        #                                    self.config.hidden_size])
        #     state = self.initial_state
        #     rnn_outputs = []
        #     for tstep, current_input in enumerate(inputs):
        #         if tstep > 0:
        #             scope.reuse_variables()
        #         RNN_H = tf.get_variable('RNN_H_matrix',
        #                                 [self.config.hidden_size,
        #                                  self.config.hidden_size])
        #         RNN_I = tf.get_variable('RNN_I_matrix',
        #                                 [2 * self.config.embed_size,
        #                                  self.config.hidden_size])
        #         RNN_b = tf.get_variable('RNN_b_bias',
        #                                 [self.config.hidden_size])
        #         state = tf.nn.sigmoid(tf.matmul(state, RNN_H) + tf.matmul(
        #             current_input, RNN_I) + RNN_b)
        #         tf.summary.histogram('hidden_weights',  RNN_H)
        #         rnn_outputs.append(state)
        #     self.final_state = rnn_outputs[-1]
        # with tf.variable_scope('RNN_dropout'):
        #     rnn_outputs = [tf.nn.dropout(x, self.dropout_placeholder) for x in
        #                    rnn_outputs]
        # return rnn_outputs
#______________________________________END_____________________________________________________________#



#________________________________________row and column,  apply next word row____________________________#
        with tf.variable_scope('InputDropout'):
            inputs = [tf.nn.dropout(x, self.dropout_placeholder) for x in
                      inputs]

        with tf.variable_scope('RNN') as scope:
            self.initial_state = tf.zeros([self.config.batch_size,
                                           self.config.hidden_size])
            state = self.initial_state
            rnn_outputs = []
            for tstep, current_input in enumerate(inputs):
                input_r, input_c, input_r_next = tf.split(current_input, 3, 1)
                if tstep > 0:
                    scope.reuse_variables()
                RNN_H = tf.get_variable('RNN_H_matrix',
                                        [self.config.hidden_size,
                                         self.config.hidden_size])
                RNN_I_r = tf.get_variable('RNN_I_r_matrix',
                                        [self.config.embed_size,
                                         self.config.hidden_size])
                RNN_I_c = tf.get_variable('RNN_I_c_matrix',
                                        [self.config.embed_size,
                                         self.config.hidden_size])
                RNN_b_r = tf.get_variable('RNN_b_r_bias',
                                        [self.config.hidden_size])
                RNN_b_c = tf.get_variable('RNN_b_c_bias',
                                        [self.config.hidden_size])
                if tstep == 0:
                    state = tf.nn.sigmoid(tf.matmul(state, RNN_H) + tf.matmul(
                        input_r, RNN_I_r) + RNN_b_r)
                    state = tf.nn.sigmoid(tf.matmul(state, RNN_H) + tf.matmul(
                        input_c, RNN_I_c) + RNN_b_c)
                    rnn_outputs.append(state)
                    state = tf.nn.sigmoid(tf.matmul(state, RNN_H) + tf.matmul(
                        input_r_next, RNN_I_r) + RNN_b_r)
                    rnn_outputs.append(state)
                else:
                    state = tf.nn.sigmoid(tf.matmul(state, RNN_H) + tf.matmul(
                        input_c, RNN_I_c) + RNN_b_c)
                    rnn_outputs.append(state)
                    state = tf.nn.sigmoid(tf.matmul(state, RNN_H) + tf.matmul(
                        input_r_next, RNN_I_r) + RNN_b_r)
                    rnn_outputs.append(state)

                tf.summary.histogram('hidden_weights',  RNN_H)
                tf.summary.histogram('hidden_state',  state)
            self.final_state = rnn_outputs[-1]
        with tf.variable_scope('RNN_dropout'):
            rnn_outputs = [tf.nn.dropout(x, self.dropout_placeholder) for x in
                           rnn_outputs]
        return rnn_outputs
#___________________________________END_______________________________________________________#

    def add_projection(self, rnn_outputs, name='project'):
        """Adds a projection layer.
        The projection layer transforms the hidden representation to a distribution
        over the vocabulary.
        Args:
          rnn_outputs: List of length num_steps, each of whose elements should be
                       a tensor of shape (batch_size, embed_size).
        Returns:
          outputs: List of length num_steps, each a tensor of shape
                   (batch_size, len(vocab)
        """


#_______________________________baseline____________________________________________________#

        # with tf.name_scope(name):
        #     U = tf.get_variable('U', [self.config.hidden_size, len(self.vocab)])
        #     b_2 = tf.get_variable('b_2', (len(self.vocab),))
        #     outputs = [tf.matmul(o, U) + b_2 for o in rnn_outputs]
        #
        #     return outputs

#__________________________________END__________________________________________________________#


# _______________________________use row and column seperatly to predict_____________________________________________#

        with tf.name_scope(name):
            U_r = tf.get_variable('U_r', [self.config.hidden_size, self.matrix_length])
            b_2_r = tf.get_variable('b_2_r', (self.matrix_length,))
            U_c = tf.get_variable('U_c', [self.config.hidden_size, self.matrix_length])
            b_2_c = tf.get_variable('b_2_c', (self.matrix_length,))
            # tf.histogram_
            assert len(rnn_outputs) % 2 == 0
            rnn_outputs_rows = rnn_outputs[::2]
            rnn_outputs_columns = rnn_outputs[1::2]
            outputs_r = [tf.matmul(i, U_r)+b_2_r for i in rnn_outputs_rows]
            outputs_c = [tf.matmul(j, U_c)+b_2_c for j in rnn_outputs_columns]
            #此时output_r,c,都是一个list,第个元素为(batch_size,matrix_length)，代表batch个预测和每个预测对应行/列的可能性。
            return outputs_r, outputs_c
            # outputs = []
            # for i in range(len(rnn_outputs)/2):
            #     outputs.append(outputs_r[i])
            #     outputs.append(outputs_c[i])
            # return outputs
        #返回的outputs是个每个元素为(batch_size, matrix_length * 2)的list

# _______________________________END____________________________________________________#


    def add_loss_op(self, output_r, output_c):
        """Adds loss ops to the computational graph.

        Hint: Use tensorflow.python.ops.seq2seq.sequence_loss to implement sequence loss.

        Args:
          output: A tensor of shape (None, self.vocab)
        Returns:
          loss: A 0-d tensor (scalar)
        """
#____________________________________base____________________________________#

        # all_ones = [tf.ones([self.config.num_steps * self.config.batch_size])]
        # cross_entropy_loss = tf.contrib.seq2seq(logits=[output],
        #                                    targets=[tf.reshape(
        #                                        tensor=self.labels_placeholder,
        #                                        shape=[-1])],
        #                                    weights=all_ones)

        # tf.add_to_collection('total_loss', cross_entropy_loss)
        # loss = tf.add_n(tf.get_collection('total_loss'))
        # self.loss_summary = tf.summary.scalar('total_loss', loss)
        # return loss
#________________________________________END______________________________________#

# ____________________rewrite for row and column loss____________________________________#
        all_ones = [tf.ones([self.config.num_steps * self.config.batch_size])]
        cross_entropy_row_loss = sequence_loss(logits=[output_r],
                                               targets=[tf.reshape(
                                                   tensor=self.labels_placeholder // self.matrix_length,
                                                   shape=[-1],)],
                                               weights = all_ones
                                               )
        cross_entropy_column_loss = sequence_loss(logits=[output_c],
                                               targets=[tf.reshape(
                                                   tensor=self.labels_placeholder % self.matrix_length,
                                                   shape=[-1], )],
                                               weights=all_ones
                                               )
        # self.print = tf.Print(_, [_], summarize=10)
        tf.add_to_collection('total_loss', cross_entropy_row_loss)
        tf.add_to_collection('total_loss', cross_entropy_column_loss)
        loss = tf.add_n(tf.get_collection('total_loss'))
        self.loss_summary = tf.summary.scalar('total_loss', loss)
        return loss



# ____________________________________END_____________________________________#
    def add_training_op(self, loss):
        """Sets up the training Ops.
        Args:
          loss: Loss tensor, from cross_entropy_loss.
        Returns:
          train_op: The Op for training.
        """
        with tf.variable_scope(tf.get_variable_scope()) as vscope:
            optimzer = tf.train.AdamOptimizer(self.config.lr)
            # optimzer.minimize函数功能：
            # 计算loss对各个变量（tf.variables)的梯度， 并更新参数
            train_op = optimzer.minimize(loss)
            tf.get_variable_scope().reuse_variables()
        return train_op



    def __init__(self, config):
        self.config = config
        self.load_data(debug=False)
        self.add_placeholders()
        self.inputs = self.add_embedding()
        self.rnn_outputs = self.add_model(self.inputs)
        # self.rnn_outputs = self.add_model_RNN(self.inputs)
        self.outputs_r, self.outputs_c = self.add_projection(self.rnn_outputs)

        # We want to check how well we correctly predict the next word
        # We cast o to float64 as there are numerical issues at hand
        # (i.e. sum(output of softmax) = 1.00000298179 and not 1)
        self.predictions_r = [tf.nn.softmax(tf.cast(o, 'float64')) for o in
                            self.outputs_r]
        self.predictions_c = [tf.nn.softmax(tf.cast(p, 'float64')) for p in
                            self.outputs_c]
        self.predictions_r = tf.reshape(tf.concat(self.predictions_r, 0), [1, -1])
        # Reshape the output into len(vocab) sized chunks - the -1 says as many as
        # needed to evenly divide
        output_r = tf.reshape(tf.concat(self.outputs_r, 1), [-1, self.matrix_length])
        output_c = tf.reshape(tf.concat(self.outputs_c, 1), [-1, self.matrix_length])
        # output = tf.reshape(tf.concat(self.outputs, 1), [-1, len(self.vocab)])
        self.calculate_loss = self.add_loss_op(output_r, output_c)
        self.train_step = self.add_training_op(self.calculate_loss)
        self.total_train_step = 0


    def run_epoch(self, session, data, train_op=None, verbose=10, writer=None):

        config = self.config
        dp = config.dropout
        is_training = 1
        if not train_op:
            train_op = tf.no_op()
            dp = 1
            is_training = 0
        total_steps = sum(
            1 for x in ptb_iterator(data, config.batch_size, config.num_steps))
        total_loss = []
        state = self.initial_state.eval()
        merged_summary = tf.summary.merge_all()
        # merged_summary = tf.summary.merge([self.loss_summary])
        for step, (x, y) in enumerate(
                ptb_iterator(data, config.batch_size, config.num_steps)):
            if is_training == 1:
                self.total_train_step += 1
            # We need to pass in the initial state and retrieve the final state to give
            # the RNN proper history
            feed = {self.input_placeholder: x,
                    self.labels_placeholder: y,
                    self.initial_state: state,
                    self.dropout_placeholder: dp}
            if step % 5 == 0:
                loss, state, _, summary_str = session.run(
                    [self.calculate_loss, self.final_state, train_op, merged_summary]
                    , feed_dict=feed)
                writer.add_summary(summary_str, self.total_train_step)
            else:
                loss, state, _ = session.run(
                    [self.calculate_loss, self.final_state, train_op],
                    feed_dict=feed)

            total_loss.append(loss)
            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : pp = {}'.format(
                    step, total_steps, np.exp(np.mean(total_loss))))
                sys.stdout.flush()
        if verbose:
            sys.stdout.write('\r')
        return np.exp(np.mean(total_loss))



def generate_text(session, model, config, starting_text='<eos>',
                  stop_length=100, stop_tokens=None, temp=1.0):
    """Generate text from the model.

          Hint: Create a feed-dictionary and use sess.run() to execute the model. Note
                that you will need to use model.initial_state as a key to feed_dict
          Hint: Fetch model.final_state and model.predictions[-1]. (You set
                model.final_state in add_model() and model.predictions is set in
                __init__)
          Hint: Store the outputs of running the model in local variables state and
                y_pred (used in the pre-implemented parts of this function.)

          Args:
            session: tf.Session() object
            model: Object of type RNNLM_Model
            config: A Config() object
            starting_text: Initial text passed to model.
          Returns:
            output: List of word idxs
          """
    with tf.name_scope('generate'):
        state = model.initial_state.eval()
        # Imagine tokens as a batch size of one, length of len(tokens[0])
        tokens = [model.vocab.encode(word) for word in starting_text.split()]
        for i in range(stop_length):
            feed = {
                model.input_placeholder: [tokens[-1:]],
                model.initial_state: state,
                model.dropout_placeholder: 1
            }
            state, y_pred = session.run(
                [model.final_state, model.predictions[-1]], feed_dict=feed
            )
            next_word_idx = sample(y_pred[0], temperature=temp)
            tokens.append(next_word_idx)
            if stop_tokens and model.vocab.decode(tokens[-1]) in stop_tokens:
                break
        output = [model.vocab.decode(word_idx) for word_idx in tokens]
        return output

def generate_sentence(session, model, config, *args, **kwargs):
    """Convenice to generate a sentence from the model."""
    return generate_text(session, model, config, *args, stop_tokens=['<eos>'],
                         **kwargs)


def test_RNNLM():
    config = Config()
    gen_config = deepcopy(config)
    gen_config.batch_size = gen_config.num_steps = 1

    # We create the training model and generative model
    with tf.variable_scope('RNNLM') as scope:
        model = RNNLM_Model(config)
        # This instructs gen_model to reuse the same variables as the model above
        scope.reuse_variables()
        # gen_model = RNNLM_Model(gen_config)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as session:
        best_val_pp = float('inf')
        best_val_epoch = 0
        session.run(init)
        train_writer = tf.summary.FileWriter('./log/temp/train', session.graph)
        valid_writer = tf.summary.FileWriter('./log/temp/valid', session.graph)
        for epoch in range(config.max_epochs):
            print('Epoch {}'.format(epoch))
            start = time.time()
            ###
            train_pp = model.run_epoch(
                session, model.encoded_train,
                train_op=model.train_step, writer=train_writer)
            valid_pp = model.run_epoch(
                session, model.encoded_valid, writer=valid_writer)
            print('Training perplexity: {}'.format(train_pp))
            print('Validation perplexity: {}'.format(valid_pp))
            if valid_pp < best_val_pp:
                best_val_pp = valid_pp
                best_val_epoch = epoch
                saver.save(session, './ptb_rnnlm.weights')
            if epoch - best_val_epoch > config.early_stopping:
                break
            print('Total time: {}'.format(time.time() - start))

        saver.restore(session, './ptb_rnnlm.weights')
        test_pp = model.run_epoch(session, model.encoded_test, writer=valid_writer)
        print('=-=' * 5)
        print('Test perplexity: {}'.format(test_pp))
        print('=-=' * 5)
        starting_text = 'whats the weather'
        # with tf.Graph().as_default():
        #     while starting_text:
        #         print(' '.join(generate_sentence(
        #             session, gen_model, gen_config, starting_text=starting_text,
        #             temp=1.0)))
        #         starting_text = input('> ')

if __name__ == "__main__":
    test_RNNLM()
