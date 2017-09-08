import numpy as np
import random
import tensorflow as tf

from config import DEFAULT_CONFIG


def build_lstm_graph_with_config(config=None):
    tf.reset_default_graph()
    lstm_graph = tf.Graph()

    if config is None:
        config = DEFAULT_CONFIG

    with lstm_graph.as_default():
        """
        The model asks for three things:
        - input: training data X
        - targets: training label y
        - learning_rate:
        """
        learning_rate = tf.placeholder(tf.float32, None, name="learning_rate")

        # Number of examples, number of input, dimension of each input
        inputs = tf.placeholder(tf.float32, [None, config.num_steps, config.input_size], name="inputs")
        targets = tf.placeholder(tf.float32, [None, config.input_size], name="targets")

        def _create_one_cell():
            lstm_cell = tf.contrib.rnn.LSTMCell(config.lstm_size, state_is_tuple=True)
            if config.keep_prob < 1.0:
                lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob)
            return lstm_cell

        cell = tf.contrib.rnn.MultiRNNCell(
            [_create_one_cell() for _ in range(config.num_layers)],
            state_is_tuple=True
        ) if config.num_layers > 1 else _create_one_cell()

        val, _ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32, scope="lilian_rnn")

        # Before transpose, val.get_shape() = (batch_size, num_steps, lstm_size)
        # After transpose, val.get_shape() = (num_steps, batch_size, lstm_size)
        val = tf.transpose(val, [1, 0, 2])

        with tf.name_scope("output_layer"):
            # last.get_shape() = (batch_size, lstm_size)
            last = tf.gather(val, int(val.get_shape()[0]) - 1, name="last_lstm_output")

            weight = tf.Variable(tf.truncated_normal([config.lstm_size, config.input_size]), name="lilian_weights")
            bias = tf.Variable(tf.constant(0.1, shape=[config.input_size]), name="lilian_biases")
            prediction = tf.matmul(last, weight) + bias

            tf.summary.histogram("last_lstm_output", last)
            tf.summary.histogram("weights", weight)
            tf.summary.histogram("biases", bias)

        with tf.name_scope("train"):
            # loss = -tf.reduce_sum(targets * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)))
            loss = tf.reduce_mean(tf.square(prediction - targets), name="loss_mse")
            optimizer = tf.train.AdamOptimizer(learning_rate)
            minimize = optimizer.minimize(loss, name="loss_mse_adam_minimize")
            tf.summary.scalar("loss_mse", loss)

        # Operators to use after restoring the model
        for op in [prediction, loss]:
            tf.add_to_collection('ops_to_restore', op)

    return lstm_graph

