# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 21:55:25 2017

@author: lankuohsing
"""

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
x = tf.placeholder(tf.float32, [None, 784])
if __name__ == '__main__':

    with tf.Session() as sess:
        x_new = tf.reshape(x, shape=[-1, 28, 28, 1])
        tf.summary.image("x", x_new, max_outputs=1)
        # Merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()
        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter("/log/",graph=tf.get_default_graph())
        choose = np.random.randint(len(mnist.test.images))
        batch_x = mnist.test.images[choose].reshape([-1, 784])

        summary = sess.run(merged_summary_op,feed_dict={x: batch_x})
        summary_writer.add_summary(summary)