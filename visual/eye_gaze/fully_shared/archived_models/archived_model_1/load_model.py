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

	Y = CuDNNLSTM(24, name = 'EG_shared_LSTM_1')(X)
	#Y = CuDNNLSTM(12, name = 'refression_LSTM_2')(Y)
	#Y = Dropout(rate = 3/9)(Y)

	#Y = Concatenate(axis = -1)([Y, X_gender])

	Y = Dense(24, activation = 'relu', name = 'EG_shared_dense_1')(Y)
	Y = Dropout(rate = 7/32)(Y)



	YR = Dense(8, activation = 'relu', name = 'EG_regression_dense_1')(Y)
	YR = Dropout(rate = 2/8)(YR)

	YR = Dense(1, activation = None, name = 'EG_regression')(YR)



	YC = Dense(8, activation = 'relu', name = 'EG_classification_dense_1')(Y)
	YC = Dropout(rate = 2/8)(YC)

	YC = Dense(5, activation = 'softmax', name = 'EG_classification')(YC)



	model = Model(inputs = X, outputs = [YR, YC])

	print("Created a new model.")

	return model