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
	#X_gender = Input(shape = (1,))

	Y = CuDNNLSTM(24, name = 'fully_shared_LSTM')(X)
	Y = Dropout(rate = 4/24)(Y)
	
	#Y = Concatenate(axis = -1)([Y, X_gender])


	YR = Dense(10, activation = 'relu')(Y)
	YR = Dropout(rate = 3/10)(YR)

	YR = Dense(1, activation = None, name = 'AU_regression')(YR)


	YC = Dense(10, activation = 'relu')(Y)
	YC = Dropout(rate = 3/10)(YC)

	YC = Dense(5, activation = 'softmax', name = 'AU_classification')(YC)



	model = Model(inputs = X, outputs = [YR, YC])

	print("Created a new model.")

	return model