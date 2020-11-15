# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPS = 1e-20


# Encoder RNN use GRU units
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


# Decoder RNN use GRU units
class DecoderRNN(nn.Module):

    def __init__(self, vocab_size, emb_size, hidden_size, dropout_rate, tied_embedding=None, enc_attention=False):
        super(DecoderRNN, self).__init__()
        self.GRU = nn.GRU(emb_size, hidden_size, dropout=dropout_rate)
        output_size = hidden_size
        if enc_attention:
            self.bilinear = nn.Bilinear(hidden_size, hidden_size, 1)
            output_size += hidden_size
        self.pre_out = nn.Linear(output_size, emb_size)

        self.out_layer = nn.Linear(emb_size, vocab_size)

        # Tied embedding means the word embedding layer weights are shared in encoder and decoder
        if tied_embedding is not None:
            self.out_layer.weight = tied_embedding.weight

    def forward(self, embedded_input, init_hidden, encoder_states=None):
        """
        One step decoder
        :param embedded_input: (batch size, embed size)
        :param init_hidden: (1, batch size, decoder hidden size)
        :return:
        """
        output, hidden = self.GRU(embedded_input.unsqueeze(0), init_hidden)
        output = output.squeeze(0)
        output_emb = self.pre_out(output)  # transform the output from hidden size to embedding size
        logits = self.out_layer(output_emb)  # calculate logits for each word
        final_output = F.softmax(logits, dim=1)
        return final_output, hidden


class Seq2Seq(nn.Module):

    def __init__(self, params, vocab, special_tokens):
        super(Seq2Seq, self).__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.special_tokens = special_tokens
        self.params = params

        self.embedding = nn.Embedding(self.vocab_size, self.params.emb_size,
                                      padding_idx=special_tokens['<PAD>']) # define word embedding layer, ignoring padding zeros
        self.encoder = EncoderRNN(params.emb_size, params.hidden_layer_units, params.dropout_rate)
        self.decoder = DecoderRNN(self.vocab_size, params.emb_size, params.hidden_layer_units, params.dropout_rate,
                                  tied_embedding=self.embedding)
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
        encoder_init_hidden = torch.zeros(1, self.params.batch_size, self.params.hidden_layer_units, device=device)
        encoder_word_embeddings = self.embedding(source_tensor)
        encoder_outputs, encoder_hidden = self.encoder(encoder_word_embeddings, encoder_init_hidden)

        # the first input of decoder RNN is set to be <SOS> token
        decoder_input_cur_step = torch.tensor([self.special_tokens['<SOS>']] * self.params.batch_size, device=device)
        # copy the final hidden states from encoder
        decoder_hidden_cur_step = encoder_hidden

        # create a tensor to record output word at each step
        output_token_idx = torch.zeros(tgt_length, self.params.batch_size, dtype=torch.long, device=device)
        total_loss = torch.tensor(0., device=device)

        # run decoder step by step
        for i in range(tgt_length):
            decoder_cur_word_embedding = self.embedding(decoder_input_cur_step) # get the word embedding
            decoder_output_cur_step, decoder_hidden_cur_step = self.decoder(
                decoder_cur_word_embedding, decoder_hidden_cur_step
            )
            _, top_idx = decoder_output_cur_step.data.topk(1) # get the most likely word idx at this step
            top_idx = top_idx.squeeze(1).detach()
            output_token_idx[i] = top_idx
            if target_tensor is not None:
                gold_standard = target_tensor[i]
            else:
                gold_standard = top_idx
            loss = self.criterion(torch.log(decoder_output_cur_step + EPS), gold_standard) # calculate the nll loss
            total_loss += loss

            # teacher forcing, use ground truth word from target tensor as input word for next step
            if self.params.teacher_forcing:
                decoder_input_cur_step = target_tensor[i]
            else:
                decoder_input_cur_step = top_idx

        return output_token_idx, total_loss

    def beam_search(self, source_tensor, beam_size):
        pass