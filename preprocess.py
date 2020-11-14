# coding=utf-8

import argparse
import pandas as pd
from text_utils.sum_dataset import SummaryDataset
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default='data/tldr_rel_17.csv')
    parser.add_argument("--article_col", default='get_tweets_column.content')
    parser.add_argument("--summary_col", default='get_tweets_column.summary')
    parser.add_argument("--test_size", default=0.2)

    args = parser.parse_args()
    raw_data = pd.read_csv(args.input_dir, encoding='utf-8')
    raw_articles = raw_data[args.article_col].astype(str)
    raw_summary = raw_data[args.summary_col].astype(str)
    print(raw_articles[:10])
    print(raw_summary[:10])

    dataset = SummaryDataset.encode_raw_article_and_summary(raw_articles, raw_summary)

    print(dataset.encoded_articles[:5])
    print(dataset.vocab)

    train_article, test_article, train_summary, test_summary = train_test_split(dataset.encoded_articles,
                                                                                dataset.encoded_summaries,
                                                                                test_size=args.test_size)
    train_dataset = SummaryDataset(vocab=dataset.vocab, encoded_articles=train_article, encoded_summaries=train_summary)
    test_dataset = SummaryDataset(vocab=dataset.vocab, encoded_articles=test_article, encoded_summaries=test_summary)

    train_dataset.save(vocab_url="data/vocab.txt",
                       encoded_articles_url="data/train_encoded_articles",
                       encoded_summaries_url="data/train_encoded_summaries")
    test_dataset.save(vocab_url="data/vocab.txt",
                      encoded_articles_url="data/test_encoded_articles",
                      encoded_summaries_url="data/test_encoded_summaries")


if __name__ == '__main__':
    main()
