"""
Load a trained model
"""
import os
import tensorflow as tf

from .config import MODEL_DIR


existing_names = [
    "SP500_lr0.05_lr_decay0.990_lstm32_step30_input1_batch200_epoch500",
]
name = "SP500_lr0.05_lr_decay0.990_lstm32_step30_input1_batch200_epoch500"


def prediction_by_trained_graph(graph_name, test_X, test_y):
    test_prediction = None
    test_loss = None

    with tf.Session() as sess:
        # Load meta graph
        graph_meta_path = os.path.join(
            MODEL_DIR, 'stock_rnn_model_{}.ckpt-499.meta'.format(graph_name))
        saver = tf.train.import_meta_graph(graph_meta_path)
        saver.restore(sess, tf.train.latest_checkpoint('_models/.'))

        graph = tf.get_default_graph()

        test_feed_dict = {
            graph.get_tensor_by_name('inputs:0'): SP500_dataset.test_X,
            graph.get_tensor_by_name('targets:0'): SP500_dataset.test_y,
            graph.get_tensor_by_name('learning_rate:0'): 0.0
        }

        ops = graph.get_collection('ops_to_restore')
        print ops

        prediction = graph.get_tensor_by_name('output_layer/add:0')
        loss = graph.get_tensor_by_name('train/loss_mse:0')
        test_prediction, test_loss = sess.run([prediction, loss], test_feed_dict)

    return test_prediction, test_loss
