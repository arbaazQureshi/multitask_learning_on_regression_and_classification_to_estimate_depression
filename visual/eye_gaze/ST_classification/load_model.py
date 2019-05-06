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

	Y = CuDNNLSTM(32, name = 'classification_LSTM_1')(X)
	#Y = CuDNNLSTM(9, name = 'classification_LSTM_2')(Y)
	#Y = Dropout(rate = 3/9)(Y)

	#Y = Concatenate(axis = -1)([Y, X_gender])

	Y = Dense(24, activation = 'relu')(Y)
	Y = Dropout(rate = 7/32)(Y)

	Y = Dense(8, activation = 'relu')(Y)
	Y = Dropout(rate = 2/8)(Y)

	Y = Dense(5, activation = 'softmax')(Y)

	model = Model(inputs = X, outputs = Y)

	print("Created a new model.")

	return model