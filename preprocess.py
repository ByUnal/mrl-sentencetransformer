import re
from vnlp import Normalizer


def remove_punctuation(text):
    return Normalizer.remove_punctuations(text)


def preprocess(text):
    # text = text.lower()
    text = remove_punctuation(text)
    text = re.sub(' +', ' ', text)
    return text
