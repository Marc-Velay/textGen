import numpy
import pandas as pd
import matplotlib.pyplot as plt
import re
import string
from gensim import corpora
from gensim.models import Phrases, Word2Vec
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import sent_tokenize
from gensim.models import KeyedVectors


def split_sentences(text):
    sents = sent_tokenize(text)
    sentences = []
    for sent in sents:
        sentence = sent.lower().split()
        filtered_sent = []
        for word in sentence:
            filtered_sent.append(re.sub('[^a-zA-Z0-9(-)]','', word))
        sentences.append(filtered_sent)
    return sentences


w2vfilename = 'w2vec/-------------------------------'
model_filename = 'w2vec/model1'
#print('loading word2vec model')
#trained_model = KeyedVectors.load_word2vec_format(w2vfilename, binary=True)
# load ascii text and covert to lowercase
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
sentences = split_sentences(raw_text)
model = Word2Vec(sentences, size=500, workers=7, window=5, min_count=1)
model.save(model_filename)
print(len(list(model.wv.vocab)))


print(model.most_similar(positive=['ring'], negative=[]))

'''
raw_text = raw_text.lower().split()
cleaned_text = []
for word in raw_text:
	cleaned_text.append(re.sub('[^a-zA-Z0-9 \n\.]','', word))
# create mapping of unique chars to integers
words = sorted(list(set(cleaned_text)))
#print(words)
word_to_int = dict((c, i) for i, c in enumerate(words))
# summarize the loaded data
n_chars = len(cleaned_text)
n_vocab = len(words)
#print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)'''
