from __future__ import print_function

import pprint

import tensorflow as tf
import tensorflow.contrib.slim as slim

from stock.dataset import load_dataset
from stock.model.rnn import LstmRNN
from stock.model.tcn import TemporalConvNet

flags = tf.app.flags
flags.DEFINE_string("model_type", "RNN", "Types of models ['TCN', 'RNN']")
flags.DEFINE_integer("stock_count", 1, "Stock count [1]")
flags.DEFINE_integer("input_size", 1, "Input size [1]")
flags.DEFINE_integer("num_steps", 30, "Num of steps [30]")
flags.DEFINE_integer("num_layers", 1, "Num of layer [1]")
flags.DEFINE_integer("hidden_size", 128, "LSTM cell size or hidden layer size in TCN. [128]")
flags.DEFINE_integer("batch_size", 32, "The size of batch images [32]")
flags.DEFINE_float("keep_prob", 0.8, "Keep probability of dropout layer. [0.8]")
flags.DEFINE_float("init_lr", 0.0005, "Initial learning rate at early stage. [0.001]")
flags.DEFINE_float("lr_decay", 0.9, "Decay rate of learning rate. [0.99]")
flags.DEFINE_integer("init_epoch", 5, "Num. of epoches considered as early stage. [5]")
flags.DEFINE_integer("max_epoch", 20, "Total training epoches. [20]")
flags.DEFINE_integer("embed_size", None, "If provided, use embedding vector of this size. [None]")
flags.DEFINE_string("stock_symbol", None, "Target stock symbol [None]")
flags.DEFINE_integer("sample_size", 4, "Number of stocks to plot during training. [4]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")

FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def main(_):
    pp.pprint(flags.FLAGS.flag_values_dict())

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    if FLAGS.stock_symbol is not None:
        FLAGS.stock_count = 1

    if FLAGS.model_type == 'RNN':
        model = LstmRNN(
            stock_count=FLAGS.stock_count,
            lstm_size=FLAGS.hidden_size,
            num_layers=FLAGS.num_layers,
            num_steps=FLAGS.num_steps,
            input_size=FLAGS.input_size,
            embed_size=FLAGS.embed_size,
        )
    elif FLAGS.model_type == 'TCN':
        model = TemporalConvNet(
            seq_len=FLAGS.num_steps,
            hidden_size=FLAGS.hidden_size,
        )
    else:
        raise ValueError(f"Unknown model type: {FLAGS.model_type}.")

    show_all_variables()
    dataset = load_dataset(
        FLAGS.input_size,
        FLAGS.num_steps,
        k=FLAGS.stock_count,
        target_symbol=FLAGS.stock_symbol,
    )

    if FLAGS.train:
        model.train(dataset, FLAGS)
    else:
        if not model.load()[0]:
            raise Exception("[!] Train a model first, then run test mode")


if __name__ == '__main__':
    tf.app.run()
