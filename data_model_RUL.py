# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 16:45:29 2018

@author: lankuohsing
"""

import numpy as np
import os
import pandas as pd
import random
import time

random.seed(time.time())


class StockDataSet(object):
    def __init__(self,
                 stock_sym,#stock_symbol
                 input_size=1,
                 num_steps=30,
                 test_ratio=0.1,#测试集占数据集的比例
                 normalized=True,
                 close_price_only=True):
        self.stock_sym = stock_sym
        self.input_size = input_size
        self.num_steps = num_steps
        self.test_ratio = test_ratio
        self.close_price_only = close_price_only
        self.normalized = normalized

        # Read csv file
        raw_df = pd.read_csv(os.path.join("data", "%s.csv" % stock_sym))

        # Merge into one sequence
        if close_price_only:
            self.raw_seq = raw_df['Close'].tolist()
        else:
            self.raw_seq = [price for tup in raw_df[['Open', 'Close']].values for price in tup]

        self.raw_seq = np.array(self.raw_seq)
        '''
        train_X.shape=[raw_seq.shape*(1-test_ratio),num_steps,input_size]  [15254,30,1]
        train_y.shape=[raw_seq.shape*(1-test_ratio),input_size]  [15254,1]
        '''
        self.train_X, self.train_y, self.test_X, self.test_y = self._prepare_data(self.raw_seq)

    def info(self):#获取股票代号为stock_sym的训练集和测试集的样本数
        return "StockDataSet [%s] train: %d test: %d" % (
            self.stock_sym, len(self.train_X), len(self.test_y))

    def _prepare_data(self, seq):
        # split into items of input_size
        seq = [np.array(seq[i * self.input_size: (i + 1) * self.input_size])
               for i in range(len(seq) // self.input_size)]

        if self.normalized:
            seq = [seq[0] / seq[0][0] - 1.0] + [
                curr / seq[i][-1] - 1.0 for i, curr in enumerate(seq[1:])]

        # split into groups of num_steps
        X = np.array([seq[i: i + self.num_steps] for i in range(len(seq) - self.num_steps)])
        y = np.array([seq[i + self.num_steps] for i in range(len(seq) - self.num_steps)])

        train_size = int(len(X) * (1.0 - self.test_ratio))
        train_X, test_X = X[:train_size], X[train_size:]
        train_y, test_y = y[:train_size], y[train_size:]
        return train_X, train_y, test_X, test_y

    def generate_one_epoch(self, batch_size):
        num_batches = int(len(self.train_X)) // batch_size#计算num_batches
        if batch_size * num_batches < len(self.train_X):#避免由于整除舍去小数后无法完全覆盖所有样本
            num_batches += 1

        batch_indices = list(range(num_batches))
        random.shuffle(batch_indices)#将序列的所有元素随机排序
        for j in batch_indices:
            batch_X = self.train_X[j * batch_size: (j + 1) * batch_size]
            batch_y = self.train_y[j * batch_size: (j + 1) * batch_size]
            assert set(map(len, batch_X)) == {self.num_steps}
            yield batch_X, batch_y
