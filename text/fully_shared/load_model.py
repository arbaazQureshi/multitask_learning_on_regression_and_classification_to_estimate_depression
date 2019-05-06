import numpy as np

from keras.models import Model, Sequential, load_model
from keras.layers import Dense, CuDNNLSTM, Input, Concatenate, Dropout, Bidirectional, TimeDistributed, Lambda, Flatten, Activation, Multiply, Add
import keras
import keras.backend as K

def load_DC_model(location=None):

	if(location != None):
		model = keras.models.load_model(location)
		print("Loaded the model.")
		return model

	X = Input(shape = (400, 512,))			#	m Tx nx
	#X_gender = Input(shape = (1,))

	Y = CuDNNLSTM(200, name = 'common_lstm_layer', return_sequences = True)(X)
	
	Y = Lambda(lambda x: K.sum(Y, axis = 1))(Y)
	
	Y = Dropout(rate = 0.3)(Y)

	#Y = Concatenate(axis = -1)([Y, X_gender])

	Y = Dense(60, activation = 'relu', name = 'DC_hidden_layer')(Y)
	Y = Dropout(rate = 0.3)(Y)
	
	Y = Dense(5, activation = 'softmax', name = 'DC_output_layer')(Y)

	model = Model(inputs = X, outputs = Y)

	print("Created a new model.")

	return model




def load_DR_model(location=None):

	if(location != None):
		model = keras.models.load_model(location)
		print("Loaded the model.")
		return model

	X = Input(shape = (400, 512,))			#	m Tx nx
	#X_gender = Input(shape = (1,))

	Y = CuDNNLSTM(200, name = 'common_lstm_layer', return_sequences = True)(X)
	
	Y = Lambda(lambda x: K.sum(Y, axis = 1))(Y)
	
	Y = Dropout(rate = 0.3)(Y)

	#Y = Concatenate(axis = -1)([Y, X_gender])

	Y = Dense(60, activation = 'relu', name = 'DR_hidden_layer')(Y)
	Y = Dropout(rate = 0.3)(Y)
	
	Y = Dense(1, activation = None, name = 'DR_output_layer')(Y)

	model = Model(inputs = X, outputs = Y)

	print("Created a new model.")

	return model


if __name__ == "__main__":
	m = load_CMU_MOSEI_model()