import numpy as np

from keras.models import Model, Sequential, load_model
from keras.layers import Dense, CuDNNLSTM, Input, Concatenate, Dropout
import keras

#from tensorflow.contrib.rnn import *


def load_model(location=None):

	if(location != None):
		model = keras.models.load_model(location)
		print("Loaded the model.")
		return model

	X = Input(shape = (10000, 6,))
	#X_gender = Input(shape = (1,))

	Y = CuDNNLSTM(7, name = 'classification_LSTM_1', return_sequences = True)(X)
	Y = CuDNNLSTM(6, name = 'classification_LSTM_2')(Y)

	Y = Dropout(rate = 3/15)(Y)

	#Y = Concatenate(axis = -1)([Y, X_gender])
	
	
	YC = Dense(5, activation = 'relu', name = 'classification_dense_1')(Y)
	YC = Dropout(rate = 3/12)(YC)

	YC = Dense(5, activation = 'softmax', name = 'HP_classification')(YC)

	model = Model(inputs = X, outputs = YC)

	print("Created a new model.")

	return model