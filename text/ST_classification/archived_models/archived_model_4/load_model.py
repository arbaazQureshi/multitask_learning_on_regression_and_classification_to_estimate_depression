import numpy as np

from keras.models import Model, Sequential, load_model
from keras.layers import Dense, CuDNNLSTM, Input, Concatenate, Dropout, Bidirectional, TimeDistributed, Lambda, Flatten, Activation, Multiply
import keras
import keras.backend as K

def load_model(location=None):

	if(location != None):
		model = keras.models.load_model(location)
		print("Loaded the model.")
		return model

	X = Input(shape = (400, 512,))

	Y = CuDNNLSTM(200, name = 'lstm_cell', return_sequences = True)(X)
	
	Y = Lambda(lambda x: K.sum(Y, axis = 1))(Y)
	Y = Dropout(rate = 0.3)(Y)

	Y = Dense(60, activation = 'relu', name = 'regressor_hidden_layer')(Y)
	Y = Dropout(rate = 0.3)(Y)
	
	Y = Dense(5, activation = 'softmax', name = 'regressor_output_layer')(Y)

	model = Model(inputs = X, outputs = Y)

	print("Created a new model.")

	return model


if __name__ == "__main__":
	m = load_model()