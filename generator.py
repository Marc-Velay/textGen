# Load LSTM network and generate text
import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.optimizers import SGD, Nadam
import re
import unidecode
from gensim import corpora
from gensim.models import Phrases, Word2Vec, KeyedVectors
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import sent_tokenize
from tqdm import tqdm


def tokenize(text):
    sents = sent_tokenize(text)
    words = []
    for sent in sents:
        sentence = sent.lower().split()
        for word in sentence:
            words.append(re.sub('[^a-zA-Z0-9(-)]','', word))
    return words


def word_from_vec(word):
	res = model_w2v.most_similar(positive=[word], negative=[])
	return res[0][0]


'''# load ascii text and covert to lowercase
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
raw_text = unidecode.unidecode(raw_text)
raw_text = raw_text.lower()#.split()
cleaned_text = raw_text
#for word in raw_text:
#	cleaned_text.append(re.sub('[^a-zA-Z0-9 \n\.]','', word))

print(cleaned_text[-10:])'''

model_filename = 'w2vec/model1'

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
r1 = r2 = r3 = None
#unidecode is for char by char processing
#raw_text = unidecode.unidecode(raw_text)
#raw_text = raw_text.lower().split()

cleaned_text = tokenize(raw_text)
# create mapping of unique chars to integers
words = sorted(list(set(cleaned_text)))
#print(words)
'''word_to_int = dict((c, i) for i, c in enumerate(words))'''

model_w2v = Word2Vec.load(model_filename)

# summarize the loaded data
n_chars = len(cleaned_text)
n_vocab = len(words)
#print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)

# prepare the dataset of input to output pairs encoded as integers
seq_length = 30
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
#print(np.array(dataX).shape)
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)
# reshape X to be [samples, time steps, features]
X = np.reshape(dataX, (n_patterns, seq_length, 500))
#del dataX
y = np.array(dataY)
del dataY

# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='tanh'))
opt = Nadam(lr=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=opt)
# load the network weights
filename = "weights/weights-improvement-02--1.5753.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer=opt)

# pick a random seed
start = np.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print("Seed:")
print("\"", ' '.join([word_from_vec(value) for value in pattern]), "\"")

# generate characters
for i in range(200):
	x = np.reshape(pattern, (1, len(pattern), 500))
	#x = x / float(n_vocab)
	prediction = model.predict(x, verbose=0)
	index = np.argmax(prediction)
	result = word_from_vec(index)
	seq_in = [word_from_vec(value) for value in pattern]
	sys.stdout.write(' ')
	sys.stdout.write(result)
	pattern.append(index)
	pattern = pattern[1:len(pattern)]
print("\nDone.")
