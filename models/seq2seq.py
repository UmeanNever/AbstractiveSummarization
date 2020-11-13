# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-20

class EncoderRNN(nn.Module):

    def __init__(self, emb_size, hidden_size, dropout_rate):
        super(EncoderRNN, self).__init__()
        self.GRU = nn.GRU(emb_size, hidden_size, dropout=dropout_rate)

    def forward(self, input_word_embeddings, init_hidden):
        """
        Multi-step encoding
        :param input_word_embeddings: (src seq len, batch size, embed size)
        :param init_hidden: (1, batch size, encoder hidden size)
        :return:
        """
        output, hidden = self.GRU(input_word_embeddings, init_hidden)
        return output, hidden


class DecoderRNN(nn.Module):

    def __init__(self, vocab_size, emb_size, hidden_size, dropout_rate, tied_embedding=None):
        super(DecoderRNN, self).__init__()
        self.GRU = nn.GRU(emb_size, hidden_size, dropout=dropout_rate)
        self.out_layer = nn.Linear(emb_size, vocab_size)
        if tied_embedding is not None:
            self.out_layer.weight = tied_embedding.weight

    def forward(self, embedded_input, init_hidden):
        """
        One step decoder
        :param embedded_input: (batch size, embed size)
        :param init_hidden: (1, batch size, decoder hidden size)
        :return:
        """
        output, hidden = self.GRU(embedded_input.unsqueeze(0), init_hidden)
        output = output.squeeze(0)
        logits = self.out_layer(output)
        final_output = F.softmax(logits)
        return final_output, hidden


class Seq2Seq(nn.Module):

    def __init__(self, params, vocab, special_tokens):
        super(Seq2Seq, self).__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.special_tokens = special_tokens
        self.params = params

        self.embedding = nn.Embedding(self.vocab_size, self.params.hidden_layer_units,
                                      padding_idx=special_tokens['<PAD>'])
        self.encoder = EncoderRNN(params.emb_size, params.hidden_layer_units, params.dropout_rate)
        self.decoder = DecoderRNN(self.vocab_size, params.emb_size, params.hidden_layer_units, params.dropout_rate)
        self.criterion = nn.NLLLoss(ignore_index=special_tokens['<PAD>'])

    def forward(self, source_tensor, target_tensor=None):
        """
        Run Seq2Seq model
        :param source_tensor: (src seq len, batch size)
        :type source_tensor: torch.Tensor
        :param target_tensor: (tgt seq len, batch size)
        :return:
        """
        if target_tensor is not None:
            tgt_length = target_tensor.size(0)
        else:
            tgt_length = self.params.max_dec_steps

        # run encoder
        encoder_init_hidden = torch.zeros(1, self.params.batch_size, self.params.hidden_layer_units)
        encoder_word_embeddings = self.embedding(source_tensor)
        encoder_outputs, encoder_hidden = self.encoder(encoder_word_embeddings, encoder_init_hidden)

        decoder_input_cur_step = torch.tensor([self.special_tokens['<SOS>']] * self.params.batch_size)
        decoder_hidden_cur_step = encoder_hidden

        output_token_idx = torch.zeros(tgt_length, self.params.batch_size, dtype=torch.long)
        total_loss = torch.tensor(0.)

        # run decoder
        for i in range(tgt_length):
            decoder_cur_word_embedding = self.embedding(decoder_input_cur_step)
            decoder_output_cur_step, decoder_hidden_cur_step = self.decoder(
                decoder_cur_word_embedding, decoder_hidden_cur_step
            )
            _, top_idx = decoder_output_cur_step.data.topk(1)
            top_idx = top_idx.squeeze(1).detach()
            output_token_idx[i] = top_idx
            gold_standard = target_tensor[i]
            loss = self.criterion(torch.log(decoder_output_cur_step + EPS), gold_standard)
            total_loss += loss

            # teacher forcing
            if self.params.teacher_forcing:
                decoder_input_cur_step = target_tensor[i]
            else:
                decoder_input_cur_step = top_idx

        return output_token_idx, total_loss



