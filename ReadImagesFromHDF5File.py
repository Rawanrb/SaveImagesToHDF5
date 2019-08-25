# Reference: http://machinelearninguru.com/deep_learning/data_preparation/hdf5/hdf5.html
import h5py
import numpy as np
from random import shuffle
from math import ceil
import matplotlib.pyplot as plt
import os


# we might need os.sep in the path in linux OS
BASE_DIR = os.path.join("..", "Downloads","dogs-vs-cats", "train")

DatasetName = "CatsVSDogs.hdf5" # path+name of the new hdf5 file
ImagesToTrain = os.path.join(BASE_DIR,"*.jpg")#path to the images you want to convert to hdf5 tables

hdf5_path = DatasetName  
dataToTrainPath = ImagesToTrain
hdf5_file = h5py.File(hdf5_path, "r")


def load_data():
	train_dataset = h5py.File(hdf5_path, "r")
	train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
	train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels
	
	test_dataset = hdf5_file
	test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
	test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

	classes = np.array(test_dataset["list_classes"][:]) # the list of classes
	
	train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
	test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
 
	return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
    
    

def training_data_number():
	hdf5_file = h5py.File(hdf5_path, "r")
	data_num = hdf5_file["train_set_x"].shape[0]
	return data_num
	



def create_batches_list():

	batch_size = 10 # number of training samples in each batch
	
	data_num = training_data_number()
	# create list of batches to shuffle the data
	batches_list = list(range(int(ceil(float(data_num) / batch_size))))
	
	shuffle(batches_list)
	
	return batches_list
	
def loop_over_batches():	

	batch_size = 10 # number of training samples in each batch
	data_num = training_data_number()
	batches_list = create_batches_list()
	hdf5_file = h5py.File(hdf5_path, "r")
	
	# loop over batches
	for n, i in enumerate(batches_list):
		i_s = i * batch_size  # index of the first image in this batch
		i_e = min([(i + 1) * batch_size, data_num])  # index of the last image in this batch

		# read batch images and remove training mean
		
		images = hdf5_file["train_set_x"][i_s:i_e, ...]


		# read labels and convert to one hot encoding 
		# https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/
		# https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
		num_class = hdf5_file ["list_classes"][:].size
		labels = hdf5_file["train_set_y"][i_s:i_e]
		labels_one_hot = np.zeros((batch_size, num_class))
		labels_one_hot[np.arange(batch_size), labels] = 1

		print(str(n+1) + "/" + str(len(batches_list)))
		print(str(labels[0]) + str(labels_one_hot[0, :]))
	
		plt.imshow(images[0])
		plt.show()

		if n == 5:  # break after 5 batches
			break

hdf5_file.close()

def main():
	loop_over_batches()
	
if __name__ == "__main__":
    main()