"""
Load a trained model
"""
import os
import tensorflow as tf

from config import MODEL_DIR


def prediction_by_trained_graph(graph_name, max_epoch, test_X, test_y):
    test_prediction = None
    test_loss = None

    with tf.Session() as sess:
        # Load meta graph
        graph_meta_path = os.path.join(
            MODEL_DIR, graph_name,
            'stock_rnn_model_{0}.ckpt-{1}.meta'.format(graph_name, max_epoch-1))
        checkpoint_path = os.path.join(MODEL_DIR, graph_name)

        saver = tf.train.import_meta_graph(graph_meta_path)
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))

        graph = tf.get_default_graph()

        test_feed_dict = {
            graph.get_tensor_by_name('inputs:0'): test_X,
            graph.get_tensor_by_name('targets:0'): test_y,
            graph.get_tensor_by_name('learning_rate:0'): 0.0
        }

        prediction = graph.get_tensor_by_name('output_layer/add:0')
        loss = graph.get_tensor_by_name('train/loss_mse:0')
        test_prediction, test_loss = sess.run([prediction, loss], test_feed_dict)

    return test_prediction, test_loss
