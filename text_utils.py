import numpy as np
import re
from gensim import corpora
from gensim.models import Phrases, Word2Vec, KeyedVectors
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import sent_tokenize
from tqdm import tqdm
import math


def tokenize(text):
    sents = sent_tokenize(text)
    words = []
    for sent in sents:
        sentence = sent.lower().split()
        for word in sentence:
            words.append(re.sub('[^a-zA-Z\-]','', word))
    return words

def split_sentences(text):
    sents = sent_tokenize(text)
    sentences = []
    for sent in sents:
        sentence = sent.lower().split()
        filtered_sent = []
        for word in sentence:
            filtered_sent.append(re.sub('[^a-zA-Z\-]','', word))
        sentences.append(filtered_sent)
    return sentences


def load_raw():
    filename = "texts/lotr.txt"
    filename2 = "texts/lotr2.txt"
    filename3 = "texts/lotr3.txt"
    r1 = open(filename).read()
    r2 = open(filename2).read()
    r3 = open(filename3).read()
    str_list = []
    str_list.append(r1)
    str_list.append(r2)
    str_list.append(r3)
    raw_text = ''.join(str_list)
    return raw_text


def load_filtered():
    filename = "texts/filtered_lotr_unk.txt"
    r1 = open(filename).read()
    raw_text = ''.join(r1)
    return raw_text


def prep_data(cleaned_text, n_chars, n_vocab, model_w2v):
    # prepare the dataset of input to output pairs encoded as integers
    seq_length = 10
    dataX = []
    dataY = []
    with tqdm(total=n_chars - seq_length) as pbar:
    	for i in range(0, n_chars - seq_length, 1):
    		pbar.update(1)
    		seq_in = cleaned_text[i:i + seq_length]
    		seq_out = cleaned_text[i + seq_length]
    		#dataX.append([word_to_int[char] for char in seq_in])
    		#dataY.append(word_to_int[seq_out])
    		dataX.append([model_w2v[word] for word in seq_in])
    		dataY.append(model_w2v[seq_out])

    return dataX, dataY, seq_length


def shuffle_in_unison(a, b):
    # courtsey http://stackoverflow.com/users/190280/josh-bleecher-snyder
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b
