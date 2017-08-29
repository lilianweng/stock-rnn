import os
import random
import re
import time
import numpy as np
import tensorflow as tf

import matplotlib
import matplotlib.pyplot as plt

from tensorflow.contrib.tensorboard.plugins import projector

matplotlib.rcParams.update({'font.size': 18})


class LstmRNN(object):
    def __init__(self, sess, stock_count,
                 lstm_size=128,
                 num_layers=1,
                 num_steps=30,
                 input_size=1,
                 keep_prob=0.8,
                 embed_size=None,
                 checkpoint_dir="checkpoints",
                 plot_dir="images"):
        """
        Construct a RNN model using LSTM cell.

        Args:
            sess:
            stock_count:
            lstm_size:
            num_layers
            num_steps:
            input_size:
            keep_prob:
            embed_size
            checkpoint_dir
        """
        self.sess = sess
        self.stock_count = stock_count

        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.input_size = input_size
        self.keep_prob = keep_prob

        self.use_embed = (embed_size is not None) and (embed_size > 0)
        self.embed_size = embed_size or 0

        self.checkpoint_dir = checkpoint_dir
        self.plot_dir = plot_dir

        self.build_graph()

    def build_graph(self):
        """
        The model asks for three things to be trained:
        - input: training data X
        - targets: training label y
        - learning_rate:
        """
        # inputs.shape = (number of examples, number of input, dimension of each input).
        self.learning_rate = tf.placeholder(tf.float32, None, name="learning_rate")
        self.symbols = tf.placeholder(tf.int32, [None,], name='stock_labels')  # mapped to an integer.

        self.inputs = tf.placeholder(tf.float32, [None, self.num_steps, self.input_size], name="inputs")
        self.targets = tf.placeholder(tf.float32, [None, self.input_size], name="targets")

        def _create_one_cell():
            lstm_cell = tf.contrib.rnn.LSTMCell(self.lstm_size, state_is_tuple=True)
            if self.keep_prob < 1.0:
                lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)
            return lstm_cell

        cell = tf.contrib.rnn.MultiRNNCell(
            [_create_one_cell() for _ in range(self.num_layers)],
            state_is_tuple=True
        ) if self.num_layers > 1 else _create_one_cell()

        val, _ = tf.nn.dynamic_rnn(cell, self.inputs, dtype=tf.float32, scope="lstm_rnn")

        # Before transpose, val.get_shape() = (batch_size, num_steps, lstm_size)
        # After transpose, val.get_shape() = (num_steps, batch_size, lstm_size)
        val = tf.transpose(val, [1, 0, 2])

        # last.get_shape() = (batch_size, lstm_size)
        last = tf.gather(val, int(val.get_shape()[0]) - 1, name="lstm_output")

        if self.embed_size > 0:
            self.embed_matrix = tf.Variable(
                tf.random_uniform([self.stock_count, self.embed_size], -1.0, 1.0),
                name="embed_matrix"
            )
            sym_embeds = tf.nn.embedding_lookup(self.embed_matrix, self.symbols)

            # After concat, last.get_shape() = (batch_size, lstm_size + embed_size)
            last = tf.concat([last, sym_embeds], axis=1, name="lstm_output_with_embed")

        ws = tf.Variable(tf.truncated_normal([
            self.lstm_size + self.embed_size, self.input_size]), name="w")
        bias = tf.Variable(tf.constant(0.1, shape=[self.input_size]), name="b")
        self.pred = tf.matmul(self.last, ws) + bias

        self.last_sum = tf.summary.histogram("lstm_output", last)
        self.w_sum = tf.summary.histogram("w", ws)
        self.b_sum = tf.summary.histogram("b", bias)
        self.pred_summ = tf.summary.histogram("pred", self.pred)

        # self.loss = -tf.reduce_sum(targets * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)))
        self.loss = tf.reduce_mean(tf.square(self.pred - self.targets), name="loss_mse")
        self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, name="adam_optim")

        self.loss_sum = tf.summary.scalar("loss_mse", self.loss)

        self.t_vars = tf.trainable_variables()
        self.saver = tf.train.Saver()

    def train(self, dataset_list, config):
        """
        Args:
            dataset_list (<StockDataSet>)
            config (tf.app.flags.FLAGS)
        """
        assert len(dataset_list) > 0
        self.merged_sum = tf.summary.merge_all()

        # Set up the logs folder
        self.writer = tf.summary.FileWriter(os.path.join("./logs", self.model_name))
        self.writer.add_graph(self.sess.graph)

        if self.use_embed:
            # Set up embedding visualization
            # Format: tensorflow/tensorboard/plugins/projector/projector_config.proto
            projector_config = projector.ProjectorConfig()

            # You can add multiple embeddings. Here we add only one.
            added_embed = projector_config.embeddings.add()
            added_embed.tensor_name = self.embed_matrix.name
            # Link this tensor to its metadata file (e.g. labels).
            added_embed.metadata_path = os.path.join("logs/metadata.tsv")

            # The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
            # read this file during startup.
            projector.visualize_embeddings(self.writer, projector_config)

        tf.global_variables_initializer().run()

        # Merged test data
        merged_test_X = []
        merged_test_y = []
        merged_test_labels = []

        for label_, d_ in enumerate(dataset_list):
            merged_test_X += list(d_.test_X)
            merged_test_y += list(d_.test_y)
            merged_test_labels += [[label_]] * len(d_.test_X)

        test_data_feed = {
            self.learning_rate: 0.0,
            self.inputs: merged_test_X,
            self.targets: merged_test_y,
            self.symbols: merged_test_labels,
        }

        global_step = 1

        num_batches = sum(len(d_.test_X) // config.batch_size for d_ in dataset_list)
        random.seed(time.time())

        # Select samples for plotting.
        sample_labels = range(4)
        sample_indices = {}
        for l in sample_labels:
            sym = dataset_list[l].stock_sym
            target_indices = np.array([
                i for i, sym_label in enumerate(merged_test_labels)
                if sym_label[0] == l])
            sample_indices[sym] = target_indices

        for epoch in xrange(config.max_epoch):
            epoch_step = 1

            learning_rate = config.init_learning_rate * (
                config.learning_rate_decay ** max(float(epoch + 1 - config.init_epoch), 0.0)
            )

            for label_, d_ in enumerate(dataset_list):
                for batch_X, batch_y in d_.generate_one_epoch(config.batch_size):
                    batch_labels = np.array([[label_]] * len(batch_X))
                    train_data_feed = {
                        self.learning_rate: learning_rate,
                        self.inputs: batch_X,
                        self.targets: batch_y,
                        self.symbols: batch_labels,
                    }
                    train_loss, _, train_merged_sum = self.sess.run(
                        [self.loss, self.optim, self.merged_sum], train_data_feed)
                    self.writer.add_summary(train_merged_sum, global_step=global_step)
                    global_step += 1
                    epoch_step += 1

                if np.mod(epoch, 20) == 0:
                    test_loss, test_pred = self.sess.run([self.loss, self.pred], test_data_feed)
                    assert len(test_pred) == len(d_.test_y)

                    print "Epoch %d [%d/%d] [learning rate: %f]: %.6f" % (
                        epoch, epoch_step, num_batches, learning_rate, test_loss)

                    # Plot samples
                    for sample_sym, indices in sample_indices.iteritems():
                        image_path = os.path.join(self.plot_dir, "{}_test_{:02d}_{:04d}.png".format(
                            sample_sym, epoch, epoch_step))
                        sample_preds = test_pred[indices]
                        sample_truth = merged_test_y[indices]
                        self.plot_samples(sample_preds, sample_truth, image_path, stock_sym=sample_sym)

        final_pred, final_loss = self.sess.run([self.pred, self.loss], test_data_feed)
        return final_pred

    @property
    def model_name(self):
        return "stock_rnn_lstm%d_step%d_input%d" % (
            self.lstm_size, self.num_steps, self.input_size)

    def save(self, step):
        model_name = self.model_name + ".model"
        model_ckpt_dir = os.path.join(self.checkpoint_dir, self.model_name)

        if not os.path.exists(model_ckpt_dir):
            os.makedirs(model_ckpt_dir)

        self.saver.save(self.sess, os.path.join(model_ckpt_dir, model_name), global_step=step)

    def load(self):
        print(" [*] Reading checkpoints...")
        model_ckpt_dir = os.path.join(self.checkpoint_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(model_ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(model_ckpt_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter

        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def plot_samples(self, preds, targets, figname, stock_sym=None):
        def _flatten(seq):
            return [x for y in seq for x in y]

        truths = _flatten(targets)
        preds = _flatten(preds)
        days = range(len(truths))

        plt.figure(figsize=(8, 6))
        plt.plot(days, truths, label='truth')
        plt.plot(days, preds, label='pred')
        plt.legend()
        plt.xlabel("day")
        plt.ylabel("normalized price")
        plt.grid(ls='--')

        if stock_sym:
            plt.title(stock_sym + " | %d days in test" % len(truths))

        plt.savefig(figname, format='png', bbox_inches='tight', transparent=True)
