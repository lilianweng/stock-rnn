"""
Run the following command to check Tensorboard:
$ tensorboard --logdir ./_logs
"""

import json
import os
import random
import tensorflow as tf

from .build_graph import default_lstm_graph
from .config import DEFAULT_CONFIG, MODEL_DIR
from .data_wrapper import StockDataSet


def load_sp500_data(config=DEFAULT_CONFIG):
    SP500_dataset = StockDataSet('SP500', config, test_ratio=0.1, close_price_only=True)
    print "traing size:", len(SP500_dataset.train_X)
    print "testing size:", len(SP500_dataset.test_X)
    print SP500_dataset.test_y


def _compute_learning_rates(config=DEFAULT_CONFIG):
    learning_rates_to_use = [
        config.init_learning_rate * (
            config.learning_rate_decay ** max(float(i + 1 - config.init_epoch), 0.0)
        ) for i in range(config.max_epoch)
    ]
    print "Middle learning rate:", learning_rates_to_use[len(learning_rates_to_use) // 2]
    return learning_rates_to_use


def train_lstm_graph(lstm_graph, stock_dataset, config=DEFAULT_CONFIG):
    """
    lstm_graph (tf.Graph)
    stock_dataset (StockDataSet)
    """
    final_prediction = []
    final_loss = None

    graph_name = "SP500_lr%.2f_lr_decay%.3f_lstm%d_step%d_input%d_batch%d_epoch%d" % (
        config.init_learning_rate, config.learning_rate_decay,
        config.lstm_size, config.num_steps,
        config.input_size, config.batch_size, config.max_epoch)

    print "Graph Name:", graph_name

    graph_saver_dir = os.path.join(MODEL_DIR, graph_name)
    if not os.path.exists(graph_saver_dir):
        os.mkdir(graph_saver_dir)

    learning_rates_to_use = _compute_learning_rates(config)
    with tf.Session(graph=lstm_graph) as sess:
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter('_logs/' + graph_name, sess.graph)
        writer.add_graph(sess.graph)

        tf.global_variables_initializer().run()

        test_data_feed = {
            inputs: stock_dataset.test_X,
            targets: stock_dataset.test_y,
            learning_rate: 0.0
        }

        for epoch_step in range(config.max_epoch):
            current_lr = learning_rates_to_use[epoch_step]

            for batch_X, batch_y in stock_dataset.generate_one_epoch(config.batch_size):
                train_data_feed = {
                    inputs: batch_X,
                    targets: batch_y,
                    learning_rate: current_lr
                }
                train_loss, _ = sess.run([loss, minimize], train_data_feed)

            if epoch_step % 10 == 0:
                test_loss, _pred, _summary = sess.run([loss, prediction, merged_summary], test_data_feed)
                assert len(_pred) == len(stock_dataset.test_y)
                print "Epoch %d [%f]:" % (epoch_step, current_lr), test_loss
                if epoch_step % 50 == 0:
                    print "Predictions:", [(
                        map(lambda x: round(x, 4), _pred[-j]),
                        map(lambda x: round(x, 4), stock_dataset.test_y[-j])
                    ) for j in range(5)]

            writer.add_summary(_summary, global_step=epoch_step)

        print "Final Results:"
        final_prediction, final_loss = sess.run([prediction, loss], test_data_feed)
        print final_prediction, final_loss

        saver = tf.train.Saver()
        saver.save(sess, os.path.join(
            graph_saver_dir, "stock_rnn_model_%s.ckpt" % graph_name), global_step=epoch_step)

    with open("final_predictions.{}.json".format(graph_name), 'w') as fout:
        fout.write(json.dumps(final_prediction.tolist()))


def main():
    sp500_dataset = load_sp500_data()
    train_lstm_graph(
        default_lstm_graph,
        sp500_dataset,
        config=DEFAULT_CONFIG
    )
