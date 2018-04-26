# Load LSTM network and generate text

import sys
import numpy as np
from keras import backend as K
import re
import unidecode
from gensim import corpora
from gensim.models import Phrases, Word2Vec, KeyedVectors
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import sent_tokenize
from tqdm import tqdm

from text_utils import *
from lstm import *



def word_from_vec(word):
	res = model_w2v.most_similar(positive=[word], negative=[])
	return res[0][0]



model_filename = 'w2vec/model9_100_occ2'
w2v_len = 100
BATCH_SIZE = 1

raw_text = load_filtered()

cleaned_text = tokenize(raw_text)[-20000:]
# create mapping of unique chars to integers
words = sorted(list(set(cleaned_text)))

model_w2v = Word2Vec.load(model_filename)


# summarize the loaded data
n_chars = len(cleaned_text)
n_vocab = len(words)
#print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)

dataX, dataY, seq_length = prep_data(cleaned_text, n_chars, n_vocab, model_w2v)
#print(np.array(dataX).shape)
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)
# reshape X to be [samples, time steps, features]
X = np.reshape(dataX, (n_patterns, seq_length, w2v_len))
#del dataX
y = np.array(dataY)
del dataY

# define the LSTM model
print('creating lstm model')
model, opt = get_lstm(X.shape, y.shape, BATCH_SIZE)

# load the network weights
filename = "weights/weights-improvement-200-0.0035.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer=opt)

# pick a random seed
start = np.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print("Seed:")
print("\"", ' '.join([word_from_vec(value) for value in pattern]), "\"")
print()

# generate characters
for i in range(50):
	x = np.reshape(np.array(pattern), (1, len(pattern), w2v_len))
	#x = x / float(n_vocab)
	prediction = model.predict(x, verbose=0, batch_size=1)
	prediction = np.reshape(prediction, (w2v_len,))
	#index = np.argmax(prediction)
	result = word_from_vec(prediction) #index
	#seq_in = [word_from_vec(np.transpose(value)) for value in pattern]
	#if not word_from_vec(prediction) == word_from_vec(pattern[-1]):
	'''print(prediction)
	print(result)
	input()'''
	sys.stdout.write(' ')
	sys.stdout.write(result)
	pattern.append(prediction)
	pattern = pattern[1:len(pattern)]
print("\nDone.")
K.clear_session()
