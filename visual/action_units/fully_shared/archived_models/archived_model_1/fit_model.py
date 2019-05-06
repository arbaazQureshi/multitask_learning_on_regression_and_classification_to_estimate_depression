from keras.models import Model, load_model

from load_model import load_model
from load_data import load_training_data, load_development_data
import keras

import numpy as np

from os import path
import os

import random

os.environ["CUDA_VISIBLE_DEVICES"]="7"





Tx = 10000
n_x = 20

training_progress = []
dev_progress = []

model = load_model()

loss_funcs = {'AU_classification' : 'categorical_crossentropy', 'AU_regression' : 'mse'}
loss_weights = {'AU_classification' : 1.0, 'AU_regression' : 1.0}
metrics = {'AU_classification' : 'accuracy', 'AU_regression' :  ['mae']}

model.compile(optimizer='adam', loss = loss_funcs, loss_weights = loss_weights, metrics = metrics)

X_train, Y_train, Y_train_class = load_training_data()
X_dev, Y_dev, Y_dev_class = load_development_data()


'''
if(path.exists('learner_params.txt')):
	learner_params = np.loadtxt('learner_params.txt')
	min_loss_dev = learner_params[0]
	min_mae = learner_params[1]
	prev_loss_dev = learner_params[4]
	loss_dev = [learner_params[2], learner_params[3]]

	increase_count = int(learner_params[5])
	current_epoch_number = int(learner_params[6])
	total_epoch_count = int(learner_params[7]) + 1
'''
	
min_MSE_dev = 10000
min_MAE_dev = 10000
min_crossentropy_dev = 10000
max_accuracy_dev = -1

current_epoch_number = 1
total_epoch_count = 1000


m = X_train.shape[0]

batch_size_list = list(range(1, m+1))




print("\n\n\n")

while(current_epoch_number < total_epoch_count):

	batch_size = m
	print("Batch size is", batch_size)
	
	hist = model.fit(X_train, [Y_train, Y_train_class], batch_size = batch_size, epochs = 1)

	total_loss_train = hist.history['loss'][0]
	MSE_train = hist.history['AU_regression_loss'][0]
	MAE_train = hist.history['AU_regression_mean_absolute_error'][0]
	crossentropy_train = hist.history['AU_classification_loss'][0]
	accuracy_train = hist.history['AU_classification_acc'][0]


	all_loss_dev = model.evaluate(X_dev, [Y_dev, Y_dev_class], batch_size = X_dev.shape[0])

	#print(loss_train)
	#print(loss_dev)

	total_loss_dev = all_loss_dev[0]
	MSE_dev = all_loss_dev[1]
	crossentropy_dev = all_loss_dev[2]
	MAE_dev = all_loss_dev[3]
	accuracy_dev = all_loss_dev[4]


	if(MSE_dev < min_MSE_dev):
		min_MSE_dev = MSE_dev

		with open('MSE_best.txt', 'w') as f:
			f.write('Min MSE:\t\t\t' + str(min_MSE_dev) + '\n')
			f.write('Corresponing MAE:\t\t' + str(MAE_dev) + '\n')
			f.write('Corresponing crossentropy:\t\t' + str(crossentropy_dev) + '\n')
			f.write('Corresponing accuracy:\t\t' + str(accuracy_dev) + '\n')
		
		model.save('best_MSE_model.h5')
		print("SAVING MSE BEST!\n\n")

	if(MAE_dev < min_MAE_dev):
		min_MAE_dev = MAE_dev

		with open('MAE_best.txt', 'w') as f:
			f.write('Min MAE:\t\t\t' + str(min_MAE_dev) + '\n')
			f.write('Corresponing MSE:\t\t' + str(MSE_dev) + '\n')
			f.write('Corresponing crossentropy:\t\t' + str(crossentropy_dev) + '\n')
			f.write('Corresponing accuracy:\t\t' + str(accuracy_dev) + '\n')
		
		model.save('best_MAE_model.h5')
		print("SAVING MAE BEST!\n\n")

	if(crossentropy_dev < min_crossentropy_dev):
		min_crossentropy_dev = crossentropy_dev

		with open('crossentropy_best.txt', 'w') as f:
			f.write('Min crossentropy:\t\t\t' + str(min_crossentropy_dev) + '\n')
			f.write('Corresponing MAE:\t\t' + str(MAE_dev) + '\n')
			f.write('Corresponing MSE:\t\t' + str(MSE_dev) + '\n')
			f.write('Corresponing accuracy:\t\t' + str(accuracy_dev) + '\n')
		
		model.save('best_crossentropy_model.h5')
		print("SAVING CROSSENTROPY BEST!\n\n")

	if(accuracy_dev > max_accuracy_dev):
		max_accuracy_dev = accuracy_dev

		with open('accuracy_best.txt', 'w') as f:
			f.write('Max accuracy:\t\t\t' + str(max_accuracy_dev) + '\n')
			f.write('Corresponing MAE:\t\t' + str(MAE_dev) + '\n')
			f.write('Corresponing MSE:\t\t' + str(MSE_dev) + '\n')
			f.write('Corresponing crossentropy:\t\t' + str(crossentropy_dev) + '\n')
		
		model.save('best_accuracy_model.h5')
		print("SAVING ACCURACY BEST!\n\n")

	training_progress.append([current_epoch_number, total_loss_train, MSE_train, MAE_train, crossentropy_train, accuracy_train])
	dev_progress.append([current_epoch_number, total_loss_dev, MSE_dev, MAE_dev, crossentropy_dev, accuracy_dev])

	np.savetxt('training_progress.csv', np.array(training_progress), fmt='%.4f', delimiter=',')
	np.savetxt('dev_progress.csv', np.array(dev_progress), fmt='%.4f', delimiter=',')
	
	current_epoch_number = current_epoch_number + 1
	print("\n\n")