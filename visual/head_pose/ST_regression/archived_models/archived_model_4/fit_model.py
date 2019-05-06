from keras.models import Model, load_model

from load_model import load_model
from load_data import load_training_data, load_development_data
import keras

import numpy as np

from os import path
import os

import random

os.environ["CUDA_VISIBLE_DEVICES"]="4,5"

training_progress = []
dev_progress = []

model = load_model()

#loss_funcs = {'HP_classification' : 'categorical_crossentropy', 'HP_regression' : 'mse'}
#loss_weights = {'HP_classification' : 1.0, 'HP_regression' : 1.0}
#metrics = {'HP_classification' : 'accuracy', 'HP_regression' :  ['mae']}

model.compile(optimizer='adagrad', loss = 'mse', metrics = ['mae'])

X_train, Y_train = load_training_data()
X_dev, Y_dev = load_development_data()
	
min_MSE_dev = 10000
min_MAE_dev = 10000
#min_crossentropy_dev = 10000
#max_accuracy_dev = -1

current_epoch_number = 1
total_epoch_count = 200

m = X_train.shape[0]
batch_size_list = list(range(1, m+1))

while(current_epoch_number <= total_epoch_count):

	batch_size = m
	print(total_epoch_count - current_epoch_number, "epochs to go.")

	hist = model.fit(X_train, Y_train, batch_size = batch_size, epochs = 1)

	MSE_train = hist.history['loss'][0]
	MAE_train = hist.history['mean_absolute_error'][0]

	all_loss_dev = model.evaluate(X_dev, Y_dev, batch_size = X_dev.shape[0])

	MSE_dev = all_loss_dev[0]
	MAE_dev = all_loss_dev[1]

	print(MSE_dev, MAE_dev, '\n')

	if(MSE_dev < min_MSE_dev):
		min_MSE_dev = MSE_dev

		with open('MSE_best.txt', 'w') as f:
			f.write('Min MSE:\t\t\t' + str(min_MSE_dev) + '\n')
			f.write('Corresponing MAE:\t\t' + str(MAE_dev) + '\n')
			#f.write('Corresponing crossentropy:\t\t' + str(crossentropy_dev) + '\n')
			#f.write('Corresponing accuracy:\t\t' + str(accuracy_dev) + '\n')
			f.write('Epoch number:\t\t' + str(current_epoch_number) + '\n')
		
		model.save('best_MSE_model.h5')
		print("SAVING MSE BEST!\n\n")

	if(MAE_dev < min_MAE_dev):
		min_MAE_dev = MAE_dev

		with open('MAE_best.txt', 'w') as f:
			f.write('Min MAE:\t\t\t' + str(min_MAE_dev) + '\n')
			f.write('Corresponing MSE:\t\t' + str(MSE_dev) + '\n')
			#f.write('Corresponing crossentropy:\t\t' + str(crossentropy_dev) + '\n')
			#f.write('Corresponing accuracy:\t\t' + str(accuracy_dev) + '\n')
			f.write('Epoch number:\t\t' + str(current_epoch_number) + '\n')
		
		model.save('best_MAE_model.h5')
		print("SAVING MAE BEST!\n\n")

	'''
	if(crossentropy_dev < min_crossentropy_dev):
		min_crossentropy_dev = crossentropy_dev

		with open('crossentropy_best.txt', 'w') as f:
			f.write('Min crossentropy:\t\t\t' + str(min_crossentropy_dev) + '\n')
			f.write('Corresponing MAE:\t\t' + str(MAE_dev) + '\n')
			f.write('Corresponing MSE:\t\t' + str(MSE_dev) + '\n')
			f.write('Corresponing accuracy:\t\t' + str(accuracy_dev) + '\n')
			f.write('Epoch number:\t\t' + str(current_epoch_number) + '\n')
		
		model.save('best_crossentropy_model.h5')
		print("SAVING CROSSENTROPY BEST!\n\n")

	if((accuracy_dev > max_accuracy_dev) or ((accuracy_dev == max_accuracy_dev) and (crossentropy_dev < min_crossentropy_dev) )):
		max_accuracy_dev = accuracy_dev

		with open('accuracy_best.txt', 'w') as f:
			f.write('Max accuracy:\t\t\t' + str(max_accuracy_dev) + '\n')
			f.write('Corresponing MAE:\t\t' + str(MAE_dev) + '\n')
			f.write('Corresponing MSE:\t\t' + str(MSE_dev) + '\n')
			f.write('Corresponing crossentropy:\t\t' + str(crossentropy_dev) + '\n')
			f.write('Epoch number:\t\t' + str(current_epoch_number) + '\n')
		
		model.save('best_accuracy_model.h5')
		print("SAVING ACCURACY BEST!\n\n")

	'''

	training_progress.append([current_epoch_number, MSE_train, MAE_train])
	dev_progress.append([current_epoch_number, MSE_dev, MAE_dev])

	np.savetxt('training_progress.csv', np.array(training_progress), fmt='%.4f', delimiter=',')
	np.savetxt('dev_progress.csv', np.array(dev_progress), fmt='%.4f', delimiter=',')
	
	current_epoch_number = current_epoch_number + 1
	print("\n\n")