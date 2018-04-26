from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM, Bidirectional
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.optimizers import SGD, Nadam

def get_lstm(xshape, yshape, BATCH_SIZE):

    model = Sequential()
    #model.add(Dense(250, batch_input_shape=(BATCH_SIZE, xshape[1], xshape[2]), activation='tanh'))
    #model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(Bidirectional(LSTM(512, return_sequences=True), batch_input_shape=(BATCH_SIZE, xshape[1], xshape[2])))
    #model.add(Dropout(0.7))
    #model.add(Dropout(0.7))
    model.add(Bidirectional(LSTM(256)))
    #model.add(Dropout(0.7))
    model.add(Dense(yshape[1], activation='tanh'))
    model.add(Dense(yshape[1], activation='tanh'))
    opt = Nadam(lr=0.00005)
    model.compile(loss='mse', optimizer=opt)

    return model,opt
