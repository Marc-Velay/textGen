import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from text_utils import *

model_filename = 'w2vec/model8_25_occ2'


raw_text = load_raw()

cleaned_text = tokenize(raw_text)

model_w2v = Word2Vec.load(model_filename)

words_known = model_w2v.wv.vocab.keys()

for r in cleaned_text:
    if r in words_known:
        appendFile = open('texts/filtered_lotr_unk.txt', 'a')
        appendFile.write(" "+r)
        appendFile.close()
    else:
        appendFile = open('texts/filtered_lotr_unk.txt', 'a')
        appendFile.write(" UNK")
        appendFile.close()
