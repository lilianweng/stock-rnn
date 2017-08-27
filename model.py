import os
import re
import numpy as np
import tensorflow as tf


class LstmRNN(object):
    def __init__(self, sess,
                 lstm_size=128,
                 num_layers=1,
                 num_steps=30,
                 input_size=1,
                 keep_prob=0.8,
                 checkpoint_dir="checkpoints"):
        """
        Construct a RNN model using LSTM cell.

        Args:
            sess:
            lstm_size:
            num_layers
            num_steps:
            input_size:
            keep_prob:
            checkpoint_dir
        """
        self.sess = sess

        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.input_size = input_size
        self.keep_prob = keep_prob

        self.checkpoint_dir = checkpoint_dir

        self.build_graph()

    def build_graph(self):
        """
        The model asks for three things to be trained:
        - input: training data X
        - targets: training label y
        - learning_rate:
        """
        # inputs.shape = (number of examples, number of input, dimension of each input).
        self.inputs = tf.placeholder(tf.float32, [None, self.num_steps, self.input_size], name="inputs")
        self.targets = tf.placeholder(tf.float32, [None, self.input_size], name="targets")
        self.learning_rate = tf.placeholder(tf.float32, None, name="learning_rate")

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

        with tf.name_scope("output_layer"):
            # last.get_shape() = (batch_size, lstm_size)
            last = tf.gather(val, int(val.get_shape()[0]) - 1, name="lstm_output")

            ws = tf.Variable(tf.truncated_normal([self.lstm_size, self.input_size]), name="w")
            bias = tf.Variable(tf.constant(0.1, shape=[self.input_size]), name="b")
            self.pred = tf.matmul(last, ws) + bias

            self.last_sum = tf.summary.histogram("lstm_output", last)
            self.w_sum = tf.summary.histogram("w", ws)
            self.b_sum = tf.summary.histogram("b", bias)
            self.pred_summ = tf.summary.histogram("pred", self.pred)

        # self.loss = -tf.reduce_sum(targets * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)))
        self.loss = tf.reduce_mean(tf.square(self.pred - self.inputs), name="loss_mse")
        self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, name="adam_optim")

        self.loss_sum = tf.summary.scalar("loss_mse", self.loss)

        self.t_vars = tf.trainable_variables()
        self.saver = tf.train.Saver()

    def train(self, dataset, config):
        """
        Args:
            dataset (StockDataSet)
            config (tf.app.flags.FLAGS)
        """

        self.merged_sum = tf.summary.merge_all()

        # Set up the logs folder
        self.writer = tf.summary.FileWriter(os.path.join("./logs", self.model_name))
        self.writer.add_graph(self.sess.graph)

        step = 1

        for epoch in xrange(config.epoch):
            learning_rate = config.init_learning_rate * (
                config.learning_rate_decay ** max(float(epoch + 1 - config.init_epoch), 0.0)
            )

            tf.global_variables_initializer().run()

            test_data_feed = {
                self.inputs: dataset.test_X,
                self.targets: dataset.test_y,
                self.learning_rate: 0.0
            }

            for batch_X, batch_y in dataset.generate_one_epoch(config.batch_size):
                train_data_feed = {
                    self.inputs: batch_X,
                    self.targets: batch_y,
                    self.learning_rate: learning_rate,
                }
                train_loss, _ = self.sess.run([self.loss, self.optim], train_data_feed)
                step += 1

                if np.mod(epoch, 10) == 0:
                    test_loss, _pred, _merged_sum = self.sess.run(
                        [self.loss, self.pred, self.merged_sum], test_data_feed)
                    assert len(_pred) == len(dataset.test_y)
                    print "Epoch %d [%f]:" % (epoch, learning_rate), test_loss
                    self.writer.add_summary(_merged_sum, global_step=epoch)

                if np.mod(step, 100) == 2:
                    self.save(self.checkpoint_dir, step)

        print "Final Results:"
        final_pred, final_loss = self.sess.run([self.pred, self.loss], test_data_feed)
        print final_pred, final_loss

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
