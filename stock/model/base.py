import os
import tensorflow as tf
from datetime import datetime


class BaseModelMixin:
    """Abstract object representing a model.
    """

    def __init__(self, model_name, saver_max_to_keep=5):
        self._saver = None
        self._saver_max_to_keep = saver_max_to_keep
        self._writer = None
        self._model_name = model_name
        self._this_model_name = None
        self._sess = None

    def scope_vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        assert len(res) > 0
        print("Variables in scope '%s'" % scope)
        for v in res:
            print("\t" + str(v))
        return res

    def save_model(self, step=None):
        # print(" [*] Saving checkpoints...")
        ckpt_file = os.path.join(self.checkpoint_dir, self.model_name)
        self.saver.save(self.sess, ckpt_file, global_step=step)

    def load_model(self):
        print(" [*] Loading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        print(self.checkpoint_dir, ckpt)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            print(ckpt_name)
            fname = os.path.join(self.checkpoint_dir, ckpt_name)
            print(fname)
            self.saver.restore(self.sess, fname)
            print(" [*] Load SUCCESS: %s" % fname)
            return True
        else:
            print(" [!] Load FAILED: %s" % self.checkpoint_dir)
            return False

    def get_path(self, dir_name):
        mod_path = os.path.join(dir_name, self.model_name)
        os.makedirs(mod_path, exist_ok=True)
        return mod_path

    @property
    def model_name(self):
        if self._this_model_name is None:
            self._this_model_name = self._model_name + '-' + datetime.now().strftime(
                '%Y-%m-%d-%H-%M-%S')
        return self._this_model_name

    @property
    def log_dir(self):
        return self.get_path("logs")

    @property
    def checkpoint_dir(self):
        return self.get_path("checkpoints")

    @property
    def image_dir(self):
        return self.get_path("images")

    @property
    def saver(self):
        if self._saver is None:
            self._saver = tf.train.Saver(max_to_keep=self._saver_max_to_keep)
        return self._saver

    @property
    def writer(self):
        if self._writer is None:
            self._writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        return self._writer

    @property
    def sess(self):
        if self._sess is None:
            config = tf.ConfigProto()

            config.intra_op_parallelism_threads = 2
            config.inter_op_parallelism_threads = 2
            self._sess = tf.Session(config=config)

        return self._sess
