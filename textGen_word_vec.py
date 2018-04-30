# Small LSTM Network to Generate Text for Alice in Wonderland
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.optimizers import SGD, Nadam
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


def tokenize(text):
    sents = sent_tokenize(text)
    words = []
    for sent in sents:
        sentence = sent.lower().split()
        for word in sentence:
            words.append(re.sub('[^a-zA-Z\-]','', word))
    return words



model_filename = 'w2vec/model9_100_occ2'
BATCH_SIZE = 128
w2v_len = 100

raw_text = load_filtered()

cleaned_text = tokenize(raw_text)
# create mapping of unique chars to integers
words = sorted(list(set(cleaned_text)))
#print(words)

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
del dataX
y = np.array(dataY)
del dataY


X, y = shuffle_in_unison(X, y)
# normalize
#X = X / float(n_vocab)
# one hot encode the output variable
#y = np_utils.to_categorical(dataY)
# define the LSTM model


print('define lstm model')
model, opt = get_lstm(X.shape, y.shape, BATCH_SIZE)
# define the checkpoint
filepath="weights/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=False, mode='min')
callbacks_list = [checkpoint]

# fit the model

new_len = X.shape[0]-(X.shape[0]%BATCH_SIZE)
X = X[:new_len]
y = y[:new_len]

history = model.fit(X, y, epochs=500, batch_size=BATCH_SIZE, callbacks=callbacks_list)

plt.figure(1)
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()


K.clear_session()
