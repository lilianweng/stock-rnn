class RNNConfig():
    input_size = 1
    num_steps = 30
    lstm_size = 128
    num_layers = 1
    keep_prob = 0.8

    batch_size = 64
    init_learning_rate = 0.001
    learning_rate_decay = 0.99
    init_epoch = 5
    max_epoch = 50

    def to_dict(self):
        dct = self.__class__.__dict__
        return {k: v for k, v in dct.iteritems() if not k.startswith('__') and not callable(v)}

    def __str__(self):
        return str(self.to_dict())

    def __repr__(self):
        return str(self.to_dict())


DEFAULT_CONFIG = RNNConfig()
print "Default configuration:", DEFAULT_CONFIG.to_dict()

DATA_DIR = "data"
LOG_DIR = "logs"
MODEL_DIR = "models"
