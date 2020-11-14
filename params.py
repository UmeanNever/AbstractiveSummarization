# coding=utf-8


class Params(object):
    def __init__(self, mode='train'):
        self.n_epoch = 50
        self.lr = 1e-4
        self.src_max_length = 64
        self.tgt_max_length = 32
        self.emb_size = 100
        self.hidden_layer_units = 128
        self.dropout_rate = 0.
        self.max_dec_steps = 32
        self.model_path_prefix = 'checkpoints/baseline'
        if mode == 'train':
            self.teacher_forcing = True
            self.batch_size = 128
        else:
            self.teacher_forcing = False
            self.batch_size = 128
