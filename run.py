# coding=utf-8

import argparse
import numpy as np
from text_utils.sum_dataset import SummaryDataset
from models.seq2seq import Seq2Seq
from tqdm import tqdm
from params import Params
from torch import optim
import torch


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
    batches = dataset.get_batch(params.batch_size, params.src_max_length, params.tgt_max_length)
    n_batches = (dataset.total_pairs - 1) // params.batch_size + 1
    model = Seq2Seq(params, dataset.vocab, dataset.SPECIAL_TOKENS)
    optimizer = optim.Adam(model.parameters(), lr=params.lr)

    for epoch_count in range(1, 1 + params.n_epoch):
        epoch_loss = 0.
        prog_bar = tqdm(range(1, n_batches + 1), desc='Epoch %d' % epoch_count)

        for batch_count, batch in enumerate(batches):
            source_tensor, target_tensor = batch
            output_tokens, batch_loss = model(source_tensor, target_tensor)
            batch_loss_value = batch_loss.item()
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss_value
            epoch_avg_loss = epoch_loss / (batch_count + 1)
            prog_bar.set_postfix(loss='%g' % epoch_avg_loss)

        # save model
        filename = "{}.{}.pt".format(params.model_path_prefix, epoch_count)
        torch.save(model, filename)


if __name__ == '__main__':
    main()
