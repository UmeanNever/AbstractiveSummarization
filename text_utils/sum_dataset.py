# coding=utf-8

from gensim.corpora import Dictionary
from text_utils.tokenize import NLTKTokenizer
import torch
import numpy as np


def build_dict_by_gensim(docs, special_tokens=None):
    if special_tokens is None:
        special_tokens = {}
    dct = Dictionary(docs)
    dct.filter_extremes(no_below=3, no_above=1.0, keep_n=None)
    dct.patch_with_special_tokens(special_tokens)
    return dct


def encode_docs_by_gensim(dct, docs, special_tokens=None):
    if special_tokens is None:
        special_tokens = {}
    encoded_docs = [np.array(dct.doc2idx(doc, special_tokens['<UNK>']), dtype=int) for doc in docs]
    return encoded_docs


class SummaryDataset(object):
    SPECIAL_TOKENS = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}

    def __init__(self, vocab, encoded_articles, encoded_summaries):
        self.vocab = vocab
        self.encoded_articles = encoded_articles
        self.encoded_summaries = encoded_summaries
        assert len(encoded_summaries) == len(encoded_articles), "X y dimension not match"
        self.total_pairs = len(encoded_summaries)

    @classmethod
    def encode_raw_article_and_summary(cls, articles, summaries):
        tokenized_articles = NLTKTokenizer.tokenize(articles)
        tokenized_summaries = NLTKTokenizer.tokenize(summaries)
        dct = build_dict_by_gensim(tokenized_articles + tokenized_summaries,
                                   special_tokens=cls.SPECIAL_TOKENS)
        encoded_articles = encode_docs_by_gensim(dct, tokenized_articles, special_tokens=cls.SPECIAL_TOKENS)
        encoded_summaries = encode_docs_by_gensim(dct, tokenized_summaries, special_tokens=cls.SPECIAL_TOKENS)
        token2id = dct.token2id
        id2token = {j: i for i, j in token2id.items()}
        vocab = [id2token[i] for i in range(len(id2token))]
        return cls(vocab, encoded_articles, encoded_summaries)

    @classmethod
    def read_encoded_article_and_summary(cls, vocab_url, encoded_articles_url, encoded_summaries_url):
        encoded_articles = np.load(encoded_articles_url, allow_pickle=True)
        encoded_summaries = np.load(encoded_summaries_url, allow_pickle=True)
        vocab = []
        with open(vocab_url, "r", encoding='utf-8') as f:
            for line in f:
                if line:
                    vocab.append(line.strip())
        return cls(vocab, encoded_articles, encoded_summaries)

    def save(self, vocab_url, encoded_articles_url, encoded_summaries_url):
        with open(vocab_url, "w", encoding='utf-8') as f:
            for word in self.vocab:
                f.write(word)
                f.write("\n")
        np.save(encoded_articles_url, np.array(self.encoded_articles))
        np.save(encoded_summaries_url, np.array(self.encoded_summaries))

    def get_batch(self, batch_size, src_max_length, tgt_max_length):
        rolling_idx = 0
        while rolling_idx < self.total_pairs:
            if rolling_idx + batch_size >= self.total_pairs:
                idx_batch = np.concatenate(
                    (np.arange(rolling_idx, self.total_pairs), range(0,
                                                                     (rolling_idx + batch_size - self.total_pairs))),
                    axis=None)
            else:
                idx_batch = np.arange(rolling_idx, rolling_idx + batch_size)
            rolling_idx = rolling_idx + batch_size
            src_tensor = torch.zeros(src_max_length, batch_size, dtype=torch.long)
            tgt_tensor = torch.zeros(tgt_max_length, batch_size, dtype=torch.long)
            for i, idx in enumerate(idx_batch):
                article = self.encoded_articles[idx]
                summary = self.encoded_summaries[idx]
                article_len = min(len(article), src_max_length - 1)
                summary_len = min(len(summary), tgt_max_length - 1)
                # print("Summary: {}".format(" ".join(str(summary[:summary_len]))))
                src_tensor[:article_len, i] = torch.as_tensor(article[:article_len], dtype=torch.long)
                src_tensor[article_len, i] = self.SPECIAL_TOKENS['<EOS>']
                tgt_tensor[:summary_len, i] = torch.as_tensor(summary[:summary_len], dtype=torch.long)
                tgt_tensor[summary_len, i] = self.SPECIAL_TOKENS['<EOS>']
                # print("Target tensor: {}".format(" ".join(str(tgt_tensor[:summary_len]))))
            yield src_tensor, tgt_tensor
