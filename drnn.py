# _*_encoding:utf-8_*_
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import fully_connected, batch_norm


class DRNN(object):
    """
    disconnect recurrent neural networks for text categorization
    """
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, hidden_size,
                 num_k, batch_size):
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.is_training = tf.placeholder(tf.bool, shape=(), name="is_training")


#       embedding layer
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0, name="W"))
            self.embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_size])
            self.embedding_init = self.W.assign(self.embedding_placeholder)
            self.embedding_chars = tf.nn.embedding_lookup(self.W, self.input_x)

        with tf.name_scope('dropout'):
            self.embedding_chars_dropout = tf.nn.dropout(self.embedding_chars, self.dropout_keep_prob)

        with tf.name_scope("DGRU_MLP"):
            self.hidden = []
            print(self.embedding_chars_dropout.shape)
            self.input = tf.pad(self.embedding_chars_dropout, [[0, 0], [num_k-1, 0], [0, 0]])
            print(self.input.shape)
            start = 0
            end = start + num_k - 1
            while end < (sequence_length+num_k-1):
                input_k = self.input[:, start:end, :]
                with tf.name_scope("gru"), tf.variable_scope('rnn') as scope:
                    cell = tf.contrib.rnn.GRUCell(hidden_size)
                    # apply dropout in hidden of rnn
                    # cell_dropout = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.dropout_keep_prob)
                    if start != 0:
                        scope.reuse_variables()
                    enconder_outputs, state = tf.nn.dynamic_rnn(cell, input_k, dtype=tf.float32)
                with tf.name_scope("dropout"):
                    state_dropout = tf.nn.dropout(state, self.dropout_keep_prob)
                with tf.name_scope("mlp"), tf.variable_scope('mlp') as scope:
                    if start != 0:
                        scope.reuse_variables()
                    # batch_norm
                    # bn_params = {
                    #     "is_training": self.is_training,
                    #     'decay': 0.99,
                    #     'updates_collections': None
                    # }
                    # mlp_output = fully_connected(state_dropout, 200, scope='mlp', normalizer_fn=batch_norm,
                    #                              normalizer_params=bn_params)
                    W = tf.get_variable("W", shape=[hidden_size, 200],
                                        initializer=tf.contrib.layers.xavier_initializer())
                    b = tf.Variable(tf.constant(0.1, shape=[200]), name="b")
                    mlp_output = tf.nn.relu(tf.nn.xw_plus_b(state_dropout, W, b), name='output')
                    self.hidden.append(mlp_output)
                self.hidden_concat = tf.concat(self.hidden, 1)
                start += 1
                end += 1

        with tf.name_scope("max-pooling"):
            hidden_reshape = tf.reshape(self.hidden_concat, [-1, sequence_length, 200])
            hidden_reshape_expand = tf.expand_dims(hidden_reshape, -1)
            pooled = tf.nn.max_pool(hidden_reshape_expand,
                                    ksize=[1, sequence_length, 1, 1],
                                    strides=[1, 1, 1, 1],
                                    padding='VALID',
                                    name='pool')
            pooled_reshape = tf.reshape(pooled, [-1, 200])

        with tf.name_scope("output"):
            W2 = tf.get_variable("W2", shape=[200, num_classes],
                                initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b2")
            self.scores = tf.nn.relu(tf.nn.xw_plus_b(pooled_reshape, W2, b2), name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


