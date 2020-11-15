# coding=utf-8
from nltk.tokenize import wordpunct_tokenize


class NLTKTokenizer(object):
    # define tokenize function
    @staticmethod
    def tokenize(docs):
        res = []
        for doc in docs:
            tokens = wordpunct_tokenize(doc.strip().lower())
            res.append(tokens)
        return res
