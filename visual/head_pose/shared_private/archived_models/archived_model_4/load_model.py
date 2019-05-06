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

	Y_reg = CuDNNLSTM(14, name = 'regression_task_LSTM_1', return_sequences = True)(X)
	Y_reg = CuDNNLSTM(11, name = 'regression_task_LSTM_2')(Y_reg)
	Y_reg = Dropout(rate = 3/11)(Y_reg)

	Y_shared = CuDNNLSTM(14, name = 'shared_LSTM_1', return_sequences = True)(X)
	Y_shared = CuDNNLSTM(11, name = 'shared_LSTM_2')(Y_shared)
	Y_shared = Dropout(rate = 3/11)(Y_shared)

	Y_class = CuDNNLSTM(14, name = 'classification_task_LSTM_1', return_sequences = True)(X)
	Y_class = CuDNNLSTM(11, name = 'classification_task_LSTM_2')(Y_class)
	Y_class = Dropout(rate = 3/11)(Y_class)




	'''
	H_reg = Concatenate(axis = -1)([Y_reg, Y_shared])

	H_reg = Dense(35, activation = 'tanh', name = 'DR_attention_hidden_layer_1')(H_reg)
	H_reg = Dropout(rate = 0.25)(H_reg)

	alpha_reg = Dense(2, activation = 'softmax', name = 'DR_attention_output_layer')(H)




	F = Lambda(lambda x : alpha[:,0:1]*Y1 + alpha[:,1:2]*Y2, name = 'DR_attention_fusion_layer')(alpha)
	'''



	Y_reg = Concatenate(axis = -1)([Y_reg, Y_shared])

	Y_reg = Dense(14, activation = 'relu', name = 'regression_dense_layer_1')(Y_reg)
	Y_reg = Dropout(rate = 3/14)(Y_reg)

	Y_reg = Dense(1, activation = None, name = 'HP_regression')(Y_reg)



	Y_class = Concatenate(axis = -1)([Y_class, Y_shared])

	Y_class = Dense(14, activation = 'relu', name = 'classification_dense_layer_1')(Y_class)
	Y_class = Dropout(rate = 3/14)(Y_class)

	Y_class = Dense(5, activation = 'softmax', name = 'HP_classification')(Y_class)



	model = Model(inputs = X, outputs = [Y_reg, Y_class])

	print("Created a new model.")

	return model