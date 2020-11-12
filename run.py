# coding=utf-8

import argparse
import numpy as np
from text_utils.sum_dataset import SummaryDataset
from models.seq2seq import Seq2Seq
from params import Params
from torch import optim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_article", default='data/encoded_articles.npy')
    parser.add_argument("--processed_summary", default='data/encoded_summaries.npy')
    parser.add_argument("--vocab", default='data/vocab.txt')
    parser.add_argument("--mode", default='train', help='train, test')

    args = parser.parse_args()
    dataset = SummaryDataset.read_encoded_article_and_summary(args.vocab,
                                                              args.processed_article,
                                                              args.processed_summary)
    print(dataset.vocab)
    params = Params()
    if args.mode == 'train':
        train(dataset, params)


def train(dataset, params):
    batches = dataset.get_batch(params.batch_size, params.max_length)
    model = Seq2Seq(params, dataset.vocab, dataset.SPECIAL_TOKENS)
    optimizer = optim.Adam(model.parameters(), lr=params.lr)

    for epoch_count in range(params.n_epoch):
        for batch in batches:
            pass


if __name__ == '__main__':
    main()
