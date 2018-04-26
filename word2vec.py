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


from text_utils import *


w2vfilename = 'w2vec/GoogleNews-vectors-negative300.bin'
model_filename = 'w2vec/model9_100_occ2'
#print('loading word2vec model')
#trained_model = KeyedVectors.load_word2vec_format(w2vfilename, binary=True)
# load ascii text and covert to lowercase
raw_text = load_raw()
raw_text = raw_text + str(' unk')
sentences = split_sentences(raw_text)
model = Word2Vec(sentences, size=100, workers=7, window=8, min_count=4, sg=1)
model.save(model_filename)
print(len(list(model.wv.vocab)))


#print(model.most_similar(positive=['ring'], negative=[]))

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
