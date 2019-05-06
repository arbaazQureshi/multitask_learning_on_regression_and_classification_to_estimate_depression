import numpy as np

from keras.models import Model, Sequential, load_model
from keras.layers import Dense, CuDNNLSTM, Input, Concatenate, Dropout
import keras


def load_model(location=None):

	if(location != None):
		model = keras.models.load_model(location)
		print("Loaded the model.")
		return model

	X = Input(shape = (10000, 20,))

	YR = CuDNNLSTM(24, name = 'fully_shared_LSTM')(X)
	YR = Dropout(rate = 4/24)(YR)

	YR = Dense(10, activation = 'relu')(YR)
	YR = Dropout(rate = 3/10)(YR)

	YR = Dense(1, activation = None, name = 'AU_regression')(YR)

	model = Model(inputs = X, outputs = YR)

	print("Created a new model.")

	return model