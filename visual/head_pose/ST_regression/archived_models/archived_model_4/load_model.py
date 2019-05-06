import numpy as np

from keras.models import Model, Sequential, load_model
from keras.layers import Dense, CuDNNLSTM, Input, Concatenate, Dropout
import keras


def load_model(location=None):

	if(location != None):
		model = keras.models.load_model(location)
		print("Loaded the model.")
		return model

	X = Input(shape = (10000, 6,))
	#X_gender = Input(shape = (1,))

	Y = CuDNNLSTM(17, name = 'regression_LSTM_1', return_sequences = True)(X)
	Y = CuDNNLSTM(14, name = 'regression_LSTM_2')(Y)

	Y = Dropout(rate = 3/14)(Y)

	#Y = Concatenate(axis = -1)([Y, X_gender])
	
	
	YR = Dense(11, activation = 'relu', name = 'regression_dense_1')(Y)
	YR = Dropout(rate = 3/11)(YR)

	YR = Dense(1, activation = None, name = 'HP_regression')(YR)

	model = Model(inputs = X, outputs = YR)

	print("Created a new model.")

	return model