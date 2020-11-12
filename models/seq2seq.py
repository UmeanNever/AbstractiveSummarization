# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderRNN(nn.Module):

    def __init__(self, emb_size, hidden_size, dropout_rate):
        super(EncoderRNN, self).__init__()
        self.GRU = nn.GRU(emb_size, hidden_size, dropout=dropout_rate)

    def forward(self, input_word_embeddings, init_hidden):
        """

        :param input_word_embeddings: (src seq len, batch size, embed size)
        :param init_hidden: (1, batch size, encoder hidden size)
        :return:
        """
        output, hidden = self.gru(input_word_embeddings, init_hidden)
        return output, hidden


class DecoderRNN(nn.Module):

    def __init__(self, vocab_size, emb_size, hidden_size, dropout_rate):
        super(DecoderRNN, self).__init__()
        self.GRU = nn.GRU(emb_size, hidden_size, dropout=dropout_rate)
        self.out_layer = nn.Linear(emb_size, vocab_size)

    def forward(self, embedded_input, init_hidden):
        """
        One step forward
        :param embedded_input: (batch size, embed size)
        :param init_hidden: (1, batch size, decoder hidden size)
        :return:
        """
        output, hidden = self.gru(embedded_input.unsqueeze(0), init_hidden)
        output = output.squeeze(0)
        logits = self.out_layer(output)
        final_output = F.softmax(logits)
        return final_output, hidden


class Seq2Seq(nn.Module):

    def __init__(self, params, vocab, special_tokens):
        super(Seq2Seq, self).__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.params = params

        self.embedding = nn.Embedding(self.vocab_size, self.params.hidden_layer_units,
                                      padding_idx=special_tokens['<PAD>'])
        self.encoder = EncoderRNN(params.emb_size, params.hidden_layer_units, params.dropout_rate)
        self.decoder = DecoderRNN(self.vocab_size, params.emb_size, params.hidden_layer_units, params.dropout_rate)

    def forward(self, source_tensor, target_tensor):
        pass
