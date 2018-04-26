from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from text_utils import *

stop_words = set(stopwords.words('english'))

raw_text = load_raw()

cleaned_text = tokenize(raw_text)
for r in cleaned_text:
    if not r in stop_words:
        appendFile = open('texts/filtered_lotr.txt', 'a')
        appendFile.write(" "+r)
        appendFile.close()
