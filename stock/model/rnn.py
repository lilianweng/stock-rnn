"""
@author: lilianweng
"""
from __future__ import print_function

import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from stock.model.base import BaseModelMixin
from stock.dataset import StockDataSet


class LstmRNN(BaseModelMixin):
    def __init__(self, stock_count: int = 1,
                 lstm_size: int = 128,
                 num_layers: int = 1,
                 num_steps: int = 30,
                 input_size: int = 1,
                 embed_size: int = None,
                 model_name: str = "rnn-test",
                 saver_max_to_keep: int = 5):
        """
        Construct a RNN model using LSTM cell.

        Args:
        - stock_count (int): num. of stocks we are going to train with.
        - lstm_size (int)
        - num_layers (int): num. of LSTM cell layers.
        - num_steps (int)
        - input_size (int)
        - keep_prob (int): (1.0 - dropout rate.) for a LSTM cell.
        - embed_size (int): length of embedding vector, only used when stock_count > 1.
        """
        super().__init__(model_name, saver_max_to_keep=saver_max_to_keep)

        self.stock_count = stock_count

        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.input_size = input_size

        self.use_embed = (embed_size is not None) and (embed_size > 0)
        self.embed_size = embed_size or -1

        self.build_graph()

    def build_graph(self):
        """
        The model asks for five things to be trained:
        - lr: learning rate
        - keep_prob: 1 - dropout rate
        - symbols: a list of stock symbols associated with each sample
        - input: training data X
        - targets: training label y
        """
        # inputs.shape = (number of examples, number of input, dimension of each input).
        self.lr = tf.placeholder(tf.float32, None, name="learning_rate")
        self.keep_prob = tf.placeholder(tf.float32, None, name="keep_prob")

        # Stock symbols are mapped to integers.
        self.labels = tf.placeholder(tf.int32, [None, 1], name='stock_labels')

        self.inputs = tf.placeholder(tf.float32, [None, self.num_steps, self.input_size],
                                     name="inputs")
        self.targets = tf.placeholder(tf.float32, [None, self.input_size], name="targets")

        def _create_one_cell():
            lstm_cell = tf.contrib.rnn.LSTMCell(self.lstm_size, state_is_tuple=True)
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)
            return lstm_cell

        cell = tf.contrib.rnn.MultiRNNCell(
            [_create_one_cell() for _ in range(self.num_layers)],
            state_is_tuple=True
        ) if self.num_layers > 1 else _create_one_cell()

        if self.embed_size > 0 and self.stock_count > 1:
            self.embed_matrix = tf.Variable(
                tf.random_uniform([self.stock_count, self.embed_size], -1.0, 1.0),
                name="embed_matrix"
            )

            # stock_label_embeds.shape = (batch_size, embedding_size)
            stacked_labels = tf.tile(self.labels, [1, self.num_steps],
                                      name='stacked_stock_labels')
            stacked_embeds = tf.nn.embedding_lookup(self.embed_matrix, stacked_labels)

            # After concat, inputs.shape = (batch_size, num_steps, input_size + embed_size)
            self.inputs_with_embed = tf.concat([self.inputs, stacked_embeds], axis=2,
                                               name="inputs_with_embed")
            self.embed_matrix_summ = tf.summary.histogram("embed_matrix", self.embed_matrix)

        else:
            self.inputs_with_embed = tf.identity(self.inputs)
            self.embed_matrix_summ = None

        print("inputs.shape:", self.inputs.shape)
        print("inputs_with_embed.shape:", self.inputs_with_embed.shape)

        # Run dynamic RNN
        val, state_ = tf.nn.dynamic_rnn(cell, self.inputs_with_embed, dtype=tf.float32,
                                        scope="dynamic_rnn")

        # Before transpose, val.get_shape() = (batch_size, num_steps, lstm_size)
        # After transpose, val.get_shape() = (num_steps, batch_size, lstm_size)
        val = tf.transpose(val, [1, 0, 2])

        last = tf.gather(val, int(val.get_shape()[0]) - 1, name="lstm_state")
        ws = tf.Variable(tf.truncated_normal([self.lstm_size, self.input_size]), name="w")
        bias = tf.Variable(tf.constant(0.1, shape=[self.input_size]), name="b")
        self.pred = tf.matmul(last, ws) + bias

        self.last_sum = tf.summary.histogram("lstm_state", last)
        self.w_sum = tf.summary.histogram("w", ws)
        self.b_sum = tf.summary.histogram("b", bias)
        self.pred_summ = tf.summary.histogram("pred", self.pred)

        # self.loss = -tf.reduce_sum(targets * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)))
        self.loss = tf.reduce_mean(tf.square(self.pred - self.targets), name="loss_mse_train")
        self.optim = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss, name="rmsprop_optim")
        # self.optim = tf.train.AdamOptimizer(self.lr).minimize(self.loss, name="adam_optim")

        # Separated from train loss.
        self.loss_test = tf.reduce_mean(tf.square(self.pred - self.targets), name="loss_mse_test")

        self.loss_summ = tf.summary.scalar("loss/mse_train", self.loss)
        self.loss_test_summ = tf.summary.scalar("loss/mse_test", self.loss_test)
        self.lr_summ = tf.summary.scalar("learning_rate", self.lr)

        self.t_vars = tf.trainable_variables()
        self.merged_summ = tf.summary.merge_all()

    def copy_embed_metadata(self):
        # Set up embedding visualization
        # Format: tensorflow/tensorboard/plugins/projector/projector_config.proto
        projector_config = projector.ProjectorConfig()

        # You can add multiple embeddings. Here we add only one.
        added_embed = projector_config.embeddings.add()
        added_embed.tensor_name = self.embed_matrix.name
        # Link this tensor to its metadata file (e.g. labels).
        shutil.copyfile(os.path.join("logs/metadata.tsv"),
                        os.path.join(self.log_dir, "metadata.tsv"))
        added_embed.metadata_path = "metadata.tsv"

        # The next line writes a projector_config.pbtxt in the LOG_DIR.
        # TensorBoard will read this file during startup.
        projector.visualize_embeddings(self.writer, projector_config)

    def train(self, dataset: StockDataSet, config: tf.app.flags.FLAGS):
        if self.use_embed:
            self.copy_embed_metadata()

        # Log the graph.
        self.writer.add_graph(self.sess.graph)
        tf.global_variables_initializer().run(session=self.sess)

        test_data_feed = {
            self.lr: 0.0,
            self.keep_prob: 1.0,
            self.inputs: dataset.test_X,
            self.targets: dataset.test_y,
            self.labels: dataset.test_labels,
        }

        global_step = 0

        # Select samples for plotting.
        sample_syms = np.random.choice(dataset.test_symbols.flatten(),
                                       min(config.sample_size, len(dataset)),
                                       replace=False)
        sample_indices = {}
        for sample_sym in sample_syms:
            target_indices = np.where(dataset.test_symbols == sample_sym)[0]
            sample_indices[sample_sym] = target_indices
        print(sample_indices)

        print("Start training for stocks:", dataset.stock_syms)
        show_every_step = len(dataset) * 200 / config.input_size

        for epoch in range(config.max_epoch):
            epoch_step = 0
            learning_rate = config.init_lr * (
                    config.lr_decay ** max(float(epoch + 1 - config.init_epoch), 0.0)
            )

            for label, batch_X, batch_y in dataset.generate_one_epoch(config.batch_size):
                global_step += 1
                epoch_step += 1
                batch_labels = np.array([[label]] * len(batch_X))
                train_data_feed = {
                    self.lr: learning_rate,
                    self.keep_prob: config.keep_prob,
                    self.inputs: batch_X,
                    self.targets: batch_y,
                    self.labels: batch_labels,
                }
                train_loss, _, train_merged_sum = self.sess.run(
                    [self.loss, self.optim, self.merged_summ], train_data_feed)
                self.writer.add_summary(train_merged_sum, global_step=global_step)

                if np.mod(global_step, show_every_step) == 1:
                    test_loss, test_pred = self.sess.run([self.loss_test, self.pred],
                                                         test_data_feed)
                    print("Step:%d [Epoch:%d] [Learning rate: %.6f] train_loss:%.6f test_loss:"
                          "%.6f" % (global_step, epoch, learning_rate, train_loss, test_loss))

                    # Plot samples
                    for sample_sym, indices in sample_indices.items():
                        sample_preds = test_pred[indices]
                        sample_truth = dataset.test_y[indices]
                        image_name = f"{sample_sym}_epoch{epoch:02d}_step{epoch_step:04d}.png"
                        self.plot_samples(
                            sample_preds, sample_truth, image_name, stock_sym=sample_sym)

            # Save model every epoch
            # self.save_model(global_step)

        final_pred, final_loss = self.sess.run([self.pred, self.loss], test_data_feed)

        # Save the final model
        self.save_model(global_step)
        return final_pred

    def evaluate(self):
        pass

    def plot_samples(self, preds, targets, image_name, stock_sym=None, multiplier=25):
        def _flatten(seq):
            return np.array([x for y in seq for x in y])

        truths = _flatten(targets)[-200:]
        preds = (_flatten(preds) * multiplier)[-200:]
        days = range(len(truths))[-200:]

        plt.figure(figsize=(12, 6))
        plt.plot(days, truths, label='truth')
        plt.plot(days, preds, label='pred')
        plt.legend(loc='upper left', frameon=False)
        plt.xlabel("day")
        plt.ylabel("normalized price")
        plt.ylim((min(truths), max(truths)))
        plt.grid(ls='--')

        if stock_sym:
            plt.title(stock_sym + " | Last %d days in test" % len(truths))

        image_path = os.path.join(self.image_dir, image_name)
        plt.savefig(image_path, format='png', bbox_inches='tight', transparent=True)
        plt.close()
