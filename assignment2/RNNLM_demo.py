#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-3-28 下午4:28
# @Author  : sadscv
# @File    : RNNLM_demo.py

import sys
import time

import numpy as np
from copy import deepcopy

from utils import calculate_perplexity, get_ptb_dataset, Vocab
from utils import ptb_iterator, sample

import tensorflow as tf

from tensorflow.contrib import rnn
from tensorflow.contrib.legacy_seq2seq import sequence_loss
# from tensorflow.python.ops.seq2seq import sequence_loss
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
  embed_size = 50
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

    def add_embedding(self):
        """Add embedding layer."""
        with tf.device('/gpu:0'):
            ### YOUR CODE HERE
            embedding = tf.get_variable('Embedding', [len(self.vocab),
                                                      self.config.embed_size],
                                        # trainable=True,
                                        )
            inputs = tf.nn.embedding_lookup(embedding, self.input_placeholder)
            # input_placeholder是2d的placeholder,[batch_size, num_steps]
            # inputs 是个3d的tensor, [batch_size, num_steps,  embed_size]
            # tf.split之后依然是3d tensor, [batch_size, 1, embed_size]
            # tf.squeeze之后变为2d tensor, [batch_size, embed_size]
            inputs = [tf.squeeze(x, [1])
                      for x in tf.split(inputs, self.config.num_steps, 1)]
            ### END YOUR CODE
            return inputs

    def add_projection(self, rnn_outputs):
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
        U = tf.get_variable('U', [self.config.hidden_size, len(self.vocab)])
        b_2 = tf.get_variable('b_2', (len(self.vocab),))
        outputs = [tf.matmul(o, U) + b_2 for o in rnn_outputs]


        return outputs

    def add_loss_op(self, output):
        """Adds loss ops to the computational graph.

        Hint: Use tensorflow.python.ops.seq2seq.sequence_loss to implement sequence loss.

        Args:
          output: A tensor of shape (None, self.vocab)
        Returns:
          loss: A 0-d tensor (scalar)
        """
        all_ones = [tf.ones([self.config.num_steps * self.config.batch_size])]
        cross_entropy_loss = sequence_loss(logits=[output],
                                           targets=[tf.reshape(
                                               tensor=self.labels_placeholder,
                                               shape=[-1])],
                                           weights=all_ones)

        tf.add_to_collection('total_loss', cross_entropy_loss)
        loss = tf.add_n(tf.get_collection('total_loss'))
        # loss = tf.contrib.seq2seq.sequence_loss()
        return loss

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
        with tf.variable_scope('InputDropout'):
            inputs = [tf.nn.dropout(x, self.dropout_placeholder) for x in
                      inputs]

        with tf.variable_scope('RNN') as scope:
            self.initial_state = tf.zeros([self.config.batch_size,
                                           self.config.hidden_size])
            state = self.initial_state
            rnn_outputs = []
            for tstep, current_input in enumerate(inputs):
                if tstep > 0:
                    scope.reuse_variables()
                RNN_H = tf.get_variable('RNN_H_matrix',
                                        [self.config.hidden_size,
                                         self.config.hidden_size])
                RNN_I = tf.get_variable('RNN_I_matrix',
                                        [self.config.embed_size,
                                         self.config.hidden_size])
                RNN_b = tf.get_variable('RNN_b_bias',
                                        [self.config.hidden_size])
                state = tf.nn.sigmoid(tf.matmul(state, RNN_H) + tf.matmul(
                    current_input, RNN_I) + RNN_b)
                rnn_outputs.append(state)
            self.final_state = rnn_outputs[-1]
        with tf.variable_scope('RNN_dropout'):
            rnn_outputs = [tf.nn.dropout(x, self.dropout_placeholder) for x in
                           rnn_outputs]
        return rnn_outputs

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

    def __init__(self, config):
        self.config = config
        self.load_data(debug=False)
        self.add_placeholders()
        self.inputs = self.add_embedding()
        self.rnn_outputs = self.add_model_RNN(self.inputs)
        self.outputs = self.add_projection(self.rnn_outputs)

        # We want to check how well we correctly predict the next word
        # We cast o to float64 as there are numerical issues at hand
        # (i.e. sum(output of softmax) = 1.00000298179 and not 1)
        self.predictions = [tf.nn.softmax(tf.cast(o, 'float64')) for o in
                            self.outputs]
        # Reshape the output into len(vocab) sized chunks - the -1 says as many as
        # needed to evenly divide
        output = tf.reshape(tf.concat(self.outputs, 1), [-1, len(self.vocab)])
        self.calculate_loss = self.add_loss_op(output)
        self.train_step = self.add_training_op(self.calculate_loss)


    def run_epoch(self, session, data, train_op=None, verbose=10):
        config = self.config
        dp = config.dropout
        if not train_op:
            train_op = tf.no_op()
            dp = 1
        total_steps = sum(
            1 for x in ptb_iterator(data, config.batch_size, config.num_steps))
        total_loss = []
        state = self.initial_state.eval()
        for step, (x, y) in enumerate(
                ptb_iterator(data, config.batch_size, config.num_steps)):
            # We need to pass in the initial state and retrieve the final state to give
            # the RNN proper history
            feed = {self.input_placeholder: x,
                    self.labels_placeholder: y,
                    self.initial_state: state,
                    self.dropout_placeholder: dp}
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
        gen_model = RNNLM_Model(gen_config)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as session:
        best_val_pp = float('inf')
        best_val_epoch = 0

        session.run(init)
        for epoch in range(config.max_epochs):
            print('Epoch {}'.format(epoch))
            start = time.time()
            ###
            train_pp = model.run_epoch(
                session, model.encoded_train,
                train_op=model.train_step)
            valid_pp = model.run_epoch(session, model.encoded_valid)
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
        test_pp = model.run_epoch(session, model.encoded_test)
        print('=-=' * 5)
        print('Test perplexity: {}'.format(test_pp))
        print('=-=' * 5)
        starting_text = 'whats the weather'
        while starting_text:
            print(' '.join(generate_sentence(
                session, gen_model, gen_config, starting_text=starting_text,
                temp=1.0)))
            starting_text = input('> ')

if __name__ == "__main__":
    test_RNNLM()
