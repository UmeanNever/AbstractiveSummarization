# coding=utf-8

import argparse
import numpy as np
from text_utils.sum_dataset import SummaryDataset
from models.seq2seq import Seq2Seq
from tqdm import tqdm
from params import Params
from torch import optim
import nltk.translate.bleu_score as bleu
import torch
from rouge import Rouge

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_article", default='data/train_encoded_articles.npy')
    parser.add_argument("--processed_summary", default='data/train_encoded_summaries.npy')
    parser.add_argument("--vocab", default='data/vocab.txt')
    parser.add_argument("--mode", default='train', help='train, test')

    args = parser.parse_args()
    dataset = SummaryDataset.read_encoded_article_and_summary(args.vocab,
                                                              args.processed_article,
                                                              args.processed_summary)
    print(dataset.vocab)

    if args.mode == 'train':
        params = Params(args.mode)
        train(dataset, params)

    if args.mode == 'test':
        params = Params(args.mode)
        test(dataset, params)


# function for training the model
def train(dataset, params):
    batches = list(dataset.get_batch(params.batch_size, params.src_max_length, params.tgt_max_length))
    n_batches = (dataset.total_pairs - 1) // params.batch_size + 1
    model = Seq2Seq(params, dataset.vocab, dataset.SPECIAL_TOKENS)  # define the model
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=params.lr)  # use ADAM optimizer

    for epoch_count in range(1, 1 + params.n_epoch):
        epoch_loss = 0.
        prog_bar = tqdm(range(1, n_batches + 1), desc='Epoch %d' % epoch_count)  # track the progress
        model.train()

        for batch_count in prog_bar:
            optimizer.zero_grad()
            batch = batches[batch_count - 1]
            source_tensor, target_tensor = batch
            source_tensor = source_tensor.to(device)
            target_tensor = target_tensor.to(device)

            # calculate output and losses
            output_tokens, batch_loss = model(source_tensor, target_tensor)

            # backward propagation
            batch_loss.backward()
            optimizer.step()

            batch_loss_value = batch_loss.item()
            epoch_loss += batch_loss_value
            epoch_avg_loss = epoch_loss / batch_count

            if batch_count % 100 == 0:
                prog_bar.set_postfix(loss='%g' % epoch_avg_loss)
                print("\n")
                print("Example Article:\n")
                print("{}\n".format(" ".join([dataset.vocab[i] for i in source_tensor[:, 0]])))
                print("Example Summary:\n")
                print("{}\n".format(" ".join([dataset.vocab[i] for i in target_tensor[:, 0]])))
                print("Output Summmary:\n")
                print("{}\n".format(" ".join([dataset.vocab[i] for i in output_tokens[:, 0]])))

        # save model
        filename = "{}.{}.pt".format(params.model_path_prefix, epoch_count)
        torch.save(model.state_dict(), filename)


# function of testing
def test(dataset, params):
    batches = list(dataset.get_batch(params.batch_size, params.src_max_length, params.tgt_max_length))
    n_batches = (dataset.total_pairs - 1) // params.batch_size + 1
    model = Seq2Seq(params, dataset.vocab, dataset.SPECIAL_TOKENS)
    model = model.to(device)

    # load model from saved checkpoint
    model.load_state_dict(torch.load(params.model_path_prefix + ".2.pt"))
    model.eval()
    rouge = Rouge()

    pred_texts = []
    target_texts = []
    source_texts = []
    loss_total = 0.
    bleu_total = 0.

    for batch_count, batch in enumerate(batches):
        source_tensor, target_tensor = batch

        # get predicted output
        with torch.no_grad():
            source_tensor = source_tensor.to(device)
            target_tensor = target_tensor.to(device)
            output_tokens, batch_loss = model.beam_search(source_tensor, params.beam_size)
        batch_loss_value = batch_loss.item()
        loss_total += batch_loss_value

        pred_text = get_raw_texts(output_tokens, vocab=dataset.vocab, special_tokens=dataset.SPECIAL_TOKENS)
        pred_texts.extend(pred_text)
        target_text = get_raw_texts(target_tensor, vocab=dataset.vocab, special_tokens=dataset.SPECIAL_TOKENS)
        target_texts.extend(target_text)
        source_text = get_raw_texts(source_tensor, vocab=dataset.vocab, special_tokens=dataset.SPECIAL_TOKENS)
        source_texts.extend(source_text)

        # calculate bleu score
        for i in range(params.batch_size):
            bleu_total += bleu.sentence_bleu([target_text[i]], pred_text[i])
        if batch_count % 100 == 0:
            print("predicting batch {} / total batch {}".format(batch_count + 1, n_batches))

    # calculate rouge score
    scores = rouge.get_scores(pred_texts, target_texts, avg=True, ignore_empty=True)
    print("Rouge scores:\n {}\n".format(scores))
    bleu_avg = bleu_total / dataset.total_pairs
    print("Bleu average scores:\n {}\n".format(bleu_avg))
    loss_average = loss_total / n_batches
    print("Negative Log Likelihood:\n {}\n".format(loss_average))

    for i in range(5):
        print("Example: {}\n".format(i + 1))
        print("Article: {}\n".format(source_texts[i]))
        print("True Summary: {}\n".format(target_texts[i]))
        print("Generated Summary: {}\n".format(pred_texts[i]))


# transfrom word indexes back to raw text
def get_raw_texts(tensors, vocab, special_tokens):
    res = []
    tensors = torch.transpose(tensors, 0, 1)
    for tensor in tensors:
        text = []
        for idx in tensor:
            if idx == special_tokens['<EOS>']:  # omit all words after the <EOS> token
                break
            text.append(vocab[idx])
        if text == ["."]:  # avoid error caused by "." when calculating rouge score
            text = [","]
        res.append(" ".join(text))
    return res


if __name__ == '__main__':
    main()
