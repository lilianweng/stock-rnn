import tensorflow as tf
import numpy as np
from stock.model.base import BaseModelMixin
from stock.dataset import StockDataSet


def causal_conv_1d(inputs,
                   filters: int,
                   kernel_size: int = 2,
                   strides: int = 1,
                   dilation_rate: int = 1,
                   use_bias: bool = True,
                   activation_fn=None):
    """
    Args:
    - inputs (tensor): bs x T x features
    - kernel_size (int): number of timesteps into the past to look (including current)
    - dilation_rate (int): number of 'holes' to leave in convolutional kernel
    - filters (int): number of separate kernels to learn
    """

    padding = (kernel_size - 1) * dilation_rate
    inputs = tf.pad(inputs, [(0, 0), (padding, 0), (0, 0)])
    return tf.layers.conv1d(
        inputs,
        filters,
        kernel_size,
        strides=strides,
        dilation_rate=dilation_rate,
        activation=activation_fn,
        use_bias=use_bias,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        bias_initializer=tf.zeros_initializer()
    )


def temporal_conv_block(inp, out_channel, kernel_size, dilation_rate,
                        scope_name, keep_prob=1.0, activate_fn='relu'):
    assert activate_fn in ['gated', 'relu']
    with tf.variable_scope(scope_name):
        if activate_fn == 'relu':
            out = causal_conv_1d(
                inp,
                out_channel,
                kernel_size=kernel_size,
                dilation_rate=dilation_rate,
            )
            out = tf.nn.relu(out)

        elif activate_fn == 'gated':
            out1 = causal_conv_1d(
                inp,
                out_channel,
                kernel_size=kernel_size,
                dilation_rate=dilation_rate,
            )
            out2 = causal_conv_1d(
                inp,
                out_channel,
                kernel_size=kernel_size,
                dilation_rate=dilation_rate,
            )
            # gated activation
            out = tf.tanh(out1) * tf.sigmoid(out2)

        else:
            raise ValueError(f"Unknown activation function: {activate_fn}")

        out = tf.nn.dropout(out, keep_prob)

    return out


class TemporalConvNet(BaseModelMixin):
    def __init__(self,
                 model_name: str = 'tcn-test',
                 seq_len: int = 300,
                 num_layers: int = 4,
                 hidden_size: int = 128,
                 keep_prob: float = 0.8,
                 saver_max_to_keep: int = 5):
        super().__init__(model_name, saver_max_to_keep=saver_max_to_keep)
        self.seq_len = seq_len
        self.kernel_size = 2
        self.num_channels = [hidden_size] * num_layers
        self.keep_prob = keep_prob

        self.build_graph()

    def build_tcn(self):
        out = self.inp
        for i, out_channel in enumerate(self.num_channels):
            dilation_rate = 2 ** i
            out = temporal_conv_block(
                out,
                out_channel,
                self.kernel_size, dilation_rate,
                f'dilation_{i}',
                keep_prob=self.keep_prob,
                activate_fn='relu'
            )

        return out

    def build_graph(self):
        self.lr = tf.placeholder(tf.float32, None, name="learning_rate")
        self.keep_prob = tf.placeholder(tf.float32, None, name="keep_prob")

        # inp.shape = (batch_size, length, channels)
        self.inp = tf.placeholder(tf.float32, [None, self.seq_len, 1], name="inputs")
        self.target = tf.placeholder(tf.float32, [None, 1], name="targets")

        tcn = self.build_tcn()

        print("tcn.shape:", tcn.shape)
        self.pred = tf.layers.dense(tcn[:, -1, :], 1, activation=None)
        self.loss = tf.losses.mean_squared_error(labels=self.target, predictions=self.pred)
        self.loss_test = tf.losses.mean_squared_error(labels=self.target, predictions=self.pred)

        self.optim = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss, name="rmsprop_optim")
        # self.optim = tf.train.AdamOptimizer(self.lr).minimize(self.loss, name="adam_optim")

        # Prepare the summary: train summary.
        self.loss_summ = tf.summary.scalar("loss/mse_train", self.loss)
        self.lr_summ = tf.summary.scalar("learning_rate", self.lr)
        self.merged_summ = tf.summary.merge([self.loss_summ, self.lr_summ])

        # Test summary
        self.loss_test_summ = tf.summary.scalar("loss/mse_test", self.loss_test)

        self.t_vars = tf.trainable_variables()

    def train(self, dataset: StockDataSet, config: tf.app.flags.FLAGS):
        """
        config should have the following values defined:
        - max_epoch
        - init_epoch
        - init_lr
        - lr_decay
        """
        self.writer.add_graph(self.sess.graph)
        tf.global_variables_initializer().run(session=self.sess)

        global_step = 0
        eval_step = 10

        for epoch in range(config.max_epoch):
            epoch_step = 0
            learning_rate = config.init_lr * (
                config.lr_decay ** max(float(epoch + 1 - config.init_epoch), 0.0)
            )

            for symbol, batch_X, batch_y in dataset.generate_one_epoch(config.batch_size):
                global_step += 1
                epoch_step += 1

                # batch_labels = np.array([[symbol]] * len(batch_X))
                train_feed = {
                    self.lr: learning_rate,
                    self.keep_prob: config.keep_prob,
                    self.inp: batch_X,
                    self.target: batch_y,
                    # self.symbols: batch_labels,
                }
                train_loss, _, train_merged_summ = self.sess.run(
                    [self.loss, self.optim, self.merged_summ], train_feed)
                self.writer.add_summary(train_merged_summ, global_step=global_step)

                # if global_step % eval_step == 0:
                test_loss = self.evaluate(dataset, global_step)
                print("Step:%d [Epoch:%d] [lr: %.6f] train_loss:%.6f test_loss:"
                      "%.6f" % (global_step, epoch, learning_rate, train_loss, test_loss))

    def evaluate(self, dataset, global_step):
        test_feed = {
            self.lr: 0.0,
            self.keep_prob: 1.0,
            self.inp: dataset.test_X,
            self.target: dataset.test_y,
        }
        test_loss, test_loss_summ = self.sess.run(
            [self.loss_test, self.loss_test_summ], test_feed)
        self.writer.add_summary(test_loss_summ, global_step=global_step)
        return test_loss
