# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 16:06:40 2018

@author: lankuohsing
"""

# In[]
'''
导入package
'''
import os
import pandas as pd
import pprint
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from data_model import StockDataSet
from model_rnn import LstmRNN
# In[]
'''
命令行参数定义
'''
flags = tf.app.flags
flags.DEFINE_integer("stock_count", 100, "Stock count [100]")
flags.DEFINE_integer("input_size", 1, "Input size [1]")
flags.DEFINE_integer("num_steps", 30, "Num of steps [30]")
flags.DEFINE_integer("num_layers", 1, "Num of layer [1]")
flags.DEFINE_integer("lstm_size", 128, "Size of one LSTM cell [128]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_float("keep_prob", 0.8, "Keep probability of dropout layer. [0.8]")
flags.DEFINE_float("init_learning_rate", 0.001, "Initial learning rate at early stage. [0.001]")
flags.DEFINE_float("learning_rate_decay", 0.99, "Decay rate of learning rate. [0.99]")
flags.DEFINE_integer("init_epoch", 5, "Num. of epoches considered as early stage. [5]")
flags.DEFINE_integer("max_epoch", 50, "Total training epoches. [50]")
flags.DEFINE_integer("embed_size", None, "If provided, use embedding vector of this size. [None]")
flags.DEFINE_string("stock_symbol", '_SP500', "Target stock symbol [None]")
flags.DEFINE_integer("sample_size", 4, "Number of stocks to plot during training. [4]")
flags.DEFINE_boolean("train", True, "True for training, False for testing [False]")
# In[]

FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()
# In[]
'''
创建日志文件夹
'''
if not os.path.exists("logs"):
    os.mkdir("logs")


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def load_sp500(input_size, num_steps, k=None, target_symbol=None, test_ratio=0.05):
    if target_symbol is not None:
        return [
            StockDataSet(
                target_symbol,#stock_symbol
                input_size=input_size,
                num_steps=num_steps,
                test_ratio=test_ratio)
        ]

    # Load metadata of s & p 500 stocks
    info = pd.read_csv("data/constituents-financials.csv")
    info = info.rename(columns={col: col.lower().replace(' ', '_') for col in info.columns})
    info['file_exists'] = info['symbol'].map(lambda x: os.path.exists("data/{}.csv".format(x)))
    print( info['file_exists'].value_counts().to_dict())

    info = info[info['file_exists'] == True].reset_index(drop=True)
    info = info.sort_values('market_cap', ascending=False).reset_index(drop=True)

    if k is not None:
        info = info.head(k)

    print( "Head of S&P 500 info:\n", info.head())

    # Generate embedding meta file
    info[['symbol', 'sector']].to_csv(os.path.join("logs/metadata.tsv"), sep='\t', index=False)

    return [
        StockDataSet(row['symbol'],
                     input_size=input_size,
                     num_steps=num_steps,
                     test_ratio=0.05)
        for _, row in info.iterrows()]
# In[]
pp.pprint(flags.FLAGS.__flags)
# In[]
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth = True

# In[]
with tf.Session(config=run_config) as sess:
    rnn_model = LstmRNN(
            sess,
            FLAGS.stock_count,
            lstm_size=FLAGS.lstm_size,
            num_layers=FLAGS.num_layers,
            num_steps=FLAGS.num_steps,
            input_size=FLAGS.input_size,
            embed_size=FLAGS.embed_size,
            )
# In[]
    show_all_variables()

# In[]
    stock_data_list = load_sp500(
            FLAGS.input_size,
            FLAGS.num_steps,
            k=FLAGS.stock_count,
            target_symbol=FLAGS.stock_symbol,
            )
    # In[]
    merged_test_X = []
    merged_test_y = []
    merged_test_labels = []
    dataset_list=stock_data_list
# In[]
    for label_, d_ in enumerate(dataset_list):
        merged_test_X += list(d_.test_X)
        merged_test_y += list(d_.test_y)
        merged_test_labels += [[label_]] * len(d_.test_X)
    # In[]
    merged_test_X = np.array(merged_test_X)
    merged_test_y = np.array(merged_test_y)
    merged_test_labels = np.array(merged_test_labels)
    # In[]
    a=len(dataset_list[0].train_X)
    b=np.array([[1,2,3],[4,5,6]])
    c=len(b)
    # In[]
    '''
    注：这里的num_batches应该覆盖所有stock_symbol的股票样本
    '''
    num_batches = sum(len(d_.train_X) for d_ in dataset_list) // FLAGS.batch_size
    # In[]
    # Select samples for plotting.
    sample_labels = range(min(FLAGS.sample_size, len(dataset_list)))#此时默认为0
    sample_indices = {}#一个字典
    for l in sample_labels:
        sym = dataset_list[l].stock_sym
        target_indices = np.array([
                                   i for i, sym_label in enumerate(merged_test_labels)
                                   if sym_label[0] == l])
        sample_indices[sym] = target_indices
    print( sample_indices)
    print(FLAGS.sample_size)
    # In[]
    global_step=0
    print( "Start training for stocks:", [d.stock_sym for d in dataset_list])
    for epoch in list(range(FLAGS.max_epoch)):
        epoch_step = 0
        learning_rate = FLAGS.init_learning_rate * (
                FLAGS.learning_rate_decay ** max(float(epoch + 1 - FLAGS.init_epoch), 0.0)
            )#早起的epoch（默认为5）之内，不对学习率进行衰减

        for label_, d_ in enumerate(dataset_list):
            for batch_X, batch_y in d_.generate_one_epoch(FLAGS.batch_size):
                global_step += 1
                epoch_step += 1
                batch_labels = np.array([[label_]] * len(batch_X))
                train_data_feed = {
                        learning_rate: learning_rate,
                        FLAGS.keep_prob: FLAGS.keep_prob,
                        FLAGS.inputs: batch_X,
                        FLAGS.targets: batch_y,
                        FLAGS.symbols: batch_labels,
                        }
