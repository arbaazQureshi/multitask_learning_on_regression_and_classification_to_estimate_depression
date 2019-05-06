import numpy as np

from keras.models import Model, Sequential, load_model
from keras.layers import Dense, CuDNNLSTM, Input, Concatenate, Dropout
import keras


def load_model(location=None):

	if(location != None):
		model = keras.models.load_model(location)
		print("Loaded the model.")
		return model

	X = Input(shape = (10000, 12,))
	#X_gender = Input(shape = (1,))

	Y = CuDNNLSTM(12, name = 'regression_LSTM_1', return_sequences = True)(X)
	Y = CuDNNLSTM(9, name = 'refression_LSTM_2')(Y)
	Y = Dropout(rate = 3/9)(Y)

	#Y = Concatenate(axis = -1)([Y, X_gender])

	Y = Dense(5, activation = 'relu')(Y)
	Y = Dropout(rate = 2/5)(Y)

	Y = Dense(1, activation = None)(Y)

	model = Model(inputs = X, outputs = Y)

	print("Created a new model.")

	return model