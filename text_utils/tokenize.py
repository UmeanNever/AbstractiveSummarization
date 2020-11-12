# coding=utf-8
from nltk.tokenize import wordpunct_tokenize


class NLTKTokenizer(object):

    @staticmethod
    def tokenize(docs):
        """
        :param docs: list of string
        :return:
        """
        res = []
        for doc in docs:
            tokens = wordpunct_tokenize(doc.strip().lower())
            res.append(tokens)
        return res
