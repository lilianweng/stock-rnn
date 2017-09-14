"""
Run the following command to check Tensorboard:
$ tensorboard --logdir ./_logs
"""
import json
import os
import sys; sys.path.append("..")
import tensorflow as tf

from build_graph import build_lstm_graph_with_config
from config import DEFAULT_CONFIG, MODEL_DIR
from data_model import StockDataSet


def load_data(stock_name, input_size, num_steps):
    stock_dataset = StockDataSet(stock_name, input_size=input_size, num_steps=num_steps,
                                 test_ratio=0.1, close_price_only=True)
    print "Train data size:", len(stock_dataset.train_X)
    print "Test data size:", len(stock_dataset.test_X)
    return stock_dataset


def _compute_learning_rates(config=DEFAULT_CONFIG):
    learning_rates_to_use = [
        config.init_learning_rate * (
            config.learning_rate_decay ** max(float(i + 1 - config.init_epoch), 0.0)
        ) for i in range(config.max_epoch)
    ]
    print "Middle learning rate:", learning_rates_to_use[len(learning_rates_to_use) // 2]
    return learning_rates_to_use


def train_lstm_graph(stock_name, lstm_graph, config=DEFAULT_CONFIG):
    """
    stock_name (str)
    lstm_graph (tf.Graph)
    """
    stock_dataset = load_data(stock_name,
                              input_size=config.input_size,
                              num_steps=config.num_steps)

    final_prediction = []
    final_loss = None

    graph_name = "%s_lr%.2f_lr_decay%.3f_lstm%d_step%d_input%d_batch%d_epoch%d" % (
        stock_name,
        config.init_learning_rate, config.learning_rate_decay,
        config.lstm_size, config.num_steps,
        config.input_size, config.batch_size, config.max_epoch)

    print "Graph Name:", graph_name

    learning_rates_to_use = _compute_learning_rates(config)
    with tf.Session(graph=lstm_graph) as sess:
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter('_logs/' + graph_name, sess.graph)
        writer.add_graph(sess.graph)

        graph = tf.get_default_graph()
        tf.global_variables_initializer().run()

        inputs = graph.get_tensor_by_name('inputs:0')
        targets = graph.get_tensor_by_name('targets:0')
        learning_rate = graph.get_tensor_by_name('learning_rate:0')

        test_data_feed = {
            inputs: stock_dataset.test_X,
            targets: stock_dataset.test_y,
            learning_rate: 0.0
        }

        loss = graph.get_tensor_by_name('train/loss_mse:0')
        minimize = graph.get_operation_by_name('train/loss_mse_adam_minimize')
        prediction = graph.get_tensor_by_name('output_layer/add:0')

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

        graph_saver_dir = os.path.join(MODEL_DIR, graph_name)
        if not os.path.exists(graph_saver_dir):
            os.mkdir(graph_saver_dir)

        saver = tf.train.Saver()
        saver.save(sess, os.path.join(
            graph_saver_dir, "stock_rnn_model_%s.ckpt" % graph_name), global_step=epoch_step)

    with open("final_predictions.{}.json".format(graph_name), 'w') as fout:
        fout.write(json.dumps(final_prediction.tolist()))


def main(config=DEFAULT_CONFIG):
    lstm_graph = build_lstm_graph_with_config(config=config)
    train_lstm_graph('SP500', lstm_graph, config=config)


if __name__ == '__main__':
    main()
