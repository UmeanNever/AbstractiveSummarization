# coding=utf-8


class Params(object):
    def __init__(self):
        self.n_epoch = 50
        self.batch_size = 64
        self.lr = 1e-3
        self.src_max_length = 64
        self.tgt_max_length = 32
        self.emb_size = 128
        self.hidden_layer_units = 128
        self.dropout_rate = 0.5
        self.max_dec_steps = 32
        self.teacher_forcing = True
        self.model_path_prefix = 'checkpoints/baseline'
