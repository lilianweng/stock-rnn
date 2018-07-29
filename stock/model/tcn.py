"""
@author: lilianweng

Check the paper on the architecture design of temporal convolutional network:

Bai, S., Kolter, J. Z., & Koltun, V. (2018). An Empirical Evaluation of Generic
Convolutional and Recurrent Networks for Sequence Modeling.

"""

import tensorflow as tf
import os
import numpy as np

from stock.dataset import StockDataSet
from stock.model.base import BaseModelMixin
from stock.utils import plot_prices


def causal_conv_1d(inp, num_channels: int,
                   kernel_size: int = 2,
                   strides: int = 1,
                   dilation_rate: int = 1,
                   use_bias: bool = True,
                   activate_fn=tf.nn.relu):
    """
    Args:
    - inp (tensor): batch_size x T x num. of channels.
    - num_channels (int): number of separate kernels to learn
    - kernel_size (int): number of timesteps into the past to look (including current)
    - dilation_rate (int): number of 'holes' to leave in convolutional kernel
    """

    padding = (kernel_size - 1) * dilation_rate
    inp = tf.pad(inp, [(0, 0), (padding, 0), (0, 0)])
    print("inputs.shape =", inp.shape)
    return tf.layers.conv1d(
        inp,
        num_channels,
        kernel_size,
        strides=strides,
        dilation_rate=dilation_rate,
        activation=activate_fn,
        use_bias=use_bias,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        bias_initializer=tf.zeros_initializer()
    )


def temporal_conv_block(inp, out_channel, kernel_size, dilation_rate,
                        scope_name, keep_prob=1.0):
    out = inp
    with tf.variable_scope(scope_name):
        out = causal_conv_1d(
            out,
            out_channel,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            activate_fn=tf.nn.relu
        )
        out = tf.contrib.layers.layer_norm(out)  # weight normalization
        out = tf.nn.dropout(out, keep_prob)

        out = causal_conv_1d(
            out,
            out_channel,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            activate_fn=tf.nn.relu
        )
        out = tf.contrib.layers.layer_norm(out)  # weight normalization
        out = tf.nn.dropout(out, keep_prob)

        out = tf.nn.relu(out + inp)

    return out


class TemporalConvNet(BaseModelMixin):
    def __init__(self, seq_len: int = 300,
                 num_layers: int = 4,
                 hidden_size: int = 128,
                 grad_clip_norm: float = 5.0,
                 weight_decay: float = 0.01,
                 model_name: str = 'tcn-test',
                 saver_max_to_keep: int = 5, ):
        super().__init__(model_name or 'tcn-test', saver_max_to_keep=saver_max_to_keep)
        self.seq_len = seq_len
        self.kernel_size = 2
        self.num_channels = [hidden_size] * num_layers
        self.weight_decay = weight_decay
        self.grad_clip_norm = grad_clip_norm

        self.build_graph()

    def build_tcn(self):
        out = self.inp
        for i, out_channel in enumerate(self.num_channels):
            dilation_rate = 2 ** i
            out = temporal_conv_block(
                out,
                out_channel,
                self.kernel_size,
                dilation_rate,
                f'temporal_block_{i}',
                keep_prob=self.keep_prob,
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
        print("pred.shape:", self.pred.shape)

        self.t_vars = tf.trainable_variables()
        print("trainable variables:", self.t_vars)

        self.reg = tf.reduce_mean([tf.nn.l2_loss(tv) for tv in self.t_vars])
        self.mse = tf.losses.mean_squared_error(labels=self.target, predictions=self.pred)
        self.mse_test = tf.losses.mean_squared_error(labels=self.target, predictions=self.pred)
        self.loss = self.mse + self.weight_decay * self.reg

        # self.optim = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss, name="rmsprop_opt")
        self.optim = tf.train.AdamOptimizer(self.lr, name="adam_optimizer")
        self.grads = self.optim.compute_gradients(self.loss, self.t_vars)
        if self.grad_clip_norm:
            self.grads = [(tf.clip_by_norm(g, self.grad_clip_norm), v) for g, v in self.grads]
        self.train_op = self.optim.apply_gradients(self.grads)

        # Prepare the summary: train summary.
        self.lr_summ = tf.summary.scalar("train/learning_rate", self.lr)
        self.mse_summ = tf.summary.scalar("train/mse", self.mse)
        self.reg_summ = tf.summary.scalar("train/reg", self.reg)
        self.loss_summ = tf.summary.scalar("train/loss", self.loss)
        self.summary = tf.summary.merge(
            [self.lr_summ, self.mse_summ, self.reg_summ, self.loss_summ])

        self.mse_test_summ = tf.summary.scalar("test/mse", self.mse_test)
        self.summary_test = tf.summary.merge([self.mse_test_summ])

    def train(self, dataset: StockDataSet, config: tf.app.flags.FLAGS):
        """
        config should have the following values defined:
        - max_epoch
        - init_epoch
        - init_lr
        - lr_decay
        """
        # Select samples for plotting.
        sample_syms = np.random.choice(dataset.test_symbols.flatten(),
                                       min(config.sample_size, len(dataset)),
                                       replace=False)
        sample_indices = {}
        for sample_sym in sample_syms:
            target_indices = np.where(dataset.test_symbols == sample_sym)[0]
            sample_indices[sample_sym] = target_indices
        print(sample_indices)

        self.writer.add_graph(self.sess.graph)
        tf.global_variables_initializer().run(session=self.sess)

        global_step = 0
        eval_step = 50

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
                val_loss_train, val_mse_train, _, val_train_summ = self.sess.run(
                    [self.loss, self.mse, self.train_op, self.summary], train_feed)
                self.writer.add_summary(val_train_summ, global_step=global_step)

                if global_step % eval_step == 0:
                    sample_img_name = f"{sample_sym}_epoch{epoch:02d}_step{epoch_step:04d}.png"
                    val_mse_test = self.evaluate(dataset, global_step, sample_indices,
                                                 sample_img_name)
                    print("Step:%d [Epoch:%d] [lr: %.6f] train_loss:%.6f train_mse:%.6f test_mse:"
                          "%.6f" % (global_step, epoch, learning_rate, val_loss_train,
                                    val_mse_train, val_mse_test))

    def evaluate(self, dataset, global_step, sample_indices, sample_img_name):
        test_feed = {
            self.lr: 0.0,
            self.keep_prob: 1.0,
            self.inp: dataset.test_X,
            self.target: dataset.test_y,
        }
        val_mse_test, val_pred_test, val_test_summary = self.sess.run(
            [self.mse_test, self.pred, self.summary_test], test_feed)
        self.writer.add_summary(val_test_summary, global_step=global_step)

        # Plot samples
        for sample_sym, indices in sample_indices.items():
            sample_preds = val_pred_test[indices]
            sample_truths = dataset.test_y[indices]
            image_path = os.path.join(self.image_dir, sample_img_name)
            plot_prices(sample_preds, sample_truths, image_path, stock_sym=None)

        return val_mse_test
