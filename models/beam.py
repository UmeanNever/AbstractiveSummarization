# coding=utf-8
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Beam(object):
    def __init__(self, beam_size, special_tokens):
        self.beam_size = beam_size
        self.special_tokens = special_tokens
        self.cur_scores = torch.zeros(beam_size, device=device)
        self.states = [torch.tensor([self.special_tokens['<SOS>']] * beam_size, device=device, dtype=torch.long)]
        self.previous_idx_history = []

    def advance(self, decoder_output):
        vocab_size = decoder_output.size(1)
        beam_scores = decoder_output + self.cur_scores.unsqueeze(1).expand_as(decoder_output)
        flat_beam_scores = beam_scores.view(-1)

        best_scores, best_score_ids = flat_beam_scores.data.topk(self.beam_size)
        self.cur_scores = best_scores

        previous_idxs = torch.floor_divide(best_score_ids, vocab_size)
        self.previous_idx_history.append(previous_idxs)
        self.states.append(best_score_ids - previous_idxs * vocab_size)

        if self.states[-1][0] == self.special_tokens['<EOS>']:
            return True
        return False

    def trace(self, idx):
        history = []
        for i in range(len(self.previous_idx_history) - 1, -1, -1):
            history.append(self.states[i + 1][idx])
            idx = self.previous_idx_history[i][idx]

        return history[::-1]
