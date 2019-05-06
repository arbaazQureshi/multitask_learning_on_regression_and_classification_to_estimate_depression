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

	Y = CuDNNLSTM(15, name = 'fully_shared_LSTM_1', return_sequences = True)(X)
	Y = CuDNNLSTM(12, name = 'fully_shared_LSTM_2')(Y)

	Y = Dropout(rate = 3/12)(Y)

	#Y = Concatenate(axis = -1)([Y, X_gender])
	
	
	YR = Dense(8, activation = 'relu', name = 'regression_dense_1')(Y)
	YR = Dropout(rate = 2/8)(YR)

	YR = Dense(1, activation = None, name = 'HP_regression')(YR)


	YC = Dense(8, activation = 'relu', name = 'classification_dense_1')(Y)
	YC = Dropout(rate = 2/8)(YC)

	YC = Dense(5, activation = 'softmax', name = 'HP_classification')(YC)

	model = Model(inputs = X, outputs = [YR, YC])

	print("Created a new model.")

	return model