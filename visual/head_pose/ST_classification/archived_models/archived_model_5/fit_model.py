from keras.models import Model, load_model

from load_model import load_model
from load_data import load_training_data, load_development_data
import keras

import numpy as np

from os import path
import os

import random

#os.environ["CUDA_VISIBLE_DEVICES"]="7"

training_progress = []
dev_progress = []

model = load_model()

model.compile(optimizer='adagrad', loss = 'categorical_crossentropy', metrics = ['accuracy'])

X_train, Y_train_class = load_training_data()
X_dev, Y_dev_class = load_development_data()
	
min_crossentropy_dev = 10000
max_accuracy_dev = -1

current_epoch_number = 1
total_epoch_count = 100

m = X_train.shape[0]
batch_size_list = list(range(1, m+1))

while(current_epoch_number <= total_epoch_count):

	batch_size = m
	print(total_epoch_count - current_epoch_number, "epochs to go.")

	hist = model.fit(X_train, Y_train_class, batch_size = batch_size, epochs = 1)

	crossentropy_train = hist.history['loss'][0]
	accuracy_train = hist.history['acc'][0]

	all_loss_dev = model.evaluate(X_dev, Y_dev_class, batch_size = X_dev.shape[0])

	crossentropy_dev = all_loss_dev[0]
	accuracy_dev = all_loss_dev[1]

	print(crossentropy_dev, accuracy_dev, '\n')

	
	if(crossentropy_dev < min_crossentropy_dev):
		min_crossentropy_dev = crossentropy_dev

		with open('crossentropy_best.txt', 'w') as f:
			f.write('Min crossentropy:\t\t\t' + str(min_crossentropy_dev) + '\n')
			f.write('Corresponing accuracy:\t\t' + str(accuracy_dev) + '\n')
			f.write('Epoch number:\t\t' + str(current_epoch_number) + '\n')
		
		model.save('best_crossentropy_model.h5')
		print("SAVING CROSSENTROPY BEST!\n\n")

	if((accuracy_dev > max_accuracy_dev) or (accuracy_dev == max_accuracy_dev and crossentropy_dev <= min_crossentropy_dev)):
		
		max_accuracy_dev = accuracy_dev

		with open('accuracy_best.txt', 'w') as f:
			f.write('Max accuracy:\t\t\t' + str(max_accuracy_dev) + '\n')
			f.write('Corresponing crossentropy:\t\t' + str(crossentropy_dev) + '\n')
			f.write('Epoch number:\t\t' + str(current_epoch_number) + '\n')
		
		model.save('best_accuracy_model.h5')
		print("SAVING ACCURACY BEST!\n\n")

	if():

		with open('accuracy_best.txt', 'w') as f:
			f.write('Max accuracy:\t\t\t' + str(max_accuracy_dev) + '\n')
			f.write('Corresponing crossentropy:\t\t' + str(crossentropy_dev) + '\n')
			f.write('Epoch number:\t\t' + str(current_epoch_number) + '\n')
		
		model.save('best_accuracy_model.h5')
		print("SAVING ACCURACY BEST!\n\n")

	training_progress.append([current_epoch_number, crossentropy_train, accuracy_train])
	dev_progress.append([current_epoch_number, crossentropy_dev, accuracy_dev])

	np.savetxt('training_progress.csv', np.array(training_progress), fmt='%.4f', delimiter=',')
	np.savetxt('dev_progress.csv', np.array(dev_progress), fmt='%.4f', delimiter=',')
	
	current_epoch_number = current_epoch_number + 1
	print("\n\n")