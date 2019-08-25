#Reference: http://machinelearninguru.com/deep_learning/data_preparation/hdf5/hdf5.html
#In this file I have not put any function because we only need to run this file for once to save the data
from random import shuffle
import glob
import numpy as np
import h5py
import cv2
import os


# we might need os.sep in the path in linux OS
BASE_DIR = os.path.join("..", "Downloads","dogs-vs-cats", "train")

DatasetName = "CatsVSDogs.hdf5" # path+name of the new hdf5 file
ImagesToTrain = os.path.join(BASE_DIR,"*.jpg")#path to the images you want to convert to hdf5 tables

hdf5Path = DatasetName  
dataToTrainPath = ImagesToTrain

classes_names = ["dog","cat"]

# read images' names which include their labels from the 'train' folder
imagesNames = glob.glob(dataToTrainPath)

labels = []
for name in imagesNames:
	if ("cat" in (name.split("/")[-1])): #to test the name of the image without the path; just taking the last part of image
		labels.append(1) # in case the image is cat give it label 1
	else:
		labels.append(0) # in case the image is dog give it label 0



# Divide the data into 60% train, 20% validation, and 20% test
trainImagesNames = imagesNames[0:int(0.6 * len(imagesNames))]
trainLabels = labels[0:int(0.6 * len(labels))]

valImagesNames = imagesNames[int(0.6 * len(imagesNames)):int(0.8 * len(imagesNames))]
valLabels = labels[int(0.6 * len(imagesNames)):int(0.8 * len(imagesNames))]

testImagesNames = imagesNames[int(0.8 * len(imagesNames)):]
testLabels = labels[int(0.8 * len(labels)):]




data_order = 'tf'  # 'th' for Theano, 'tf' for Tensorflow

# check the order of data and chose proper data shape to save images
if data_order == 'th':
    trainShape = (len(trainImagesNames), 3, 224, 224)
    valShape = (len(valImagesNames), 3, 224, 224)
    testShape = (len(testImagesNames), 3, 224, 224)
elif data_order == 'tf':
    trainShape = (len(trainImagesNames), 224, 224, 3)
    valShape = (len(valImagesNames), 224, 224, 3)
    testShape = (len(testImagesNames), 224, 224, 3)

# open a hdf5 file and create earrays
hdf5File = h5py.File(hdf5Path, mode='w')

hdf5File.create_dataset("train_set_x", trainShape, np.int8)
hdf5File.create_dataset("val_set_x", valShape, np.int8)
hdf5File.create_dataset("test_set_x", testShape, np.int8)

hdf5File.create_dataset("train_mean", trainShape[1:], np.float32)


#we have two classes here dog and cat
# to save the names of classes as string ex. dog and cat
dt = h5py.special_dtype(vlen=str)
hdf5File.create_dataset("list_classes", (1,len(classes_names)), dtype=dt) 
hdf5File["list_classes"][...]=classes_names

hdf5File.create_dataset("train_set_y", (len(trainImagesNames),), np.int8)
hdf5File["train_set_y"][...] = trainLabels
hdf5File.create_dataset("val_set_y", (len(valImagesNames),), np.int8)
hdf5File["val_set_y"][...] = valLabels
hdf5File.create_dataset("test_set_y", (len(testImagesNames),), np.int8)
hdf5File["test_set_y"][...] = testLabels



# a numpy array to save the mean of the images
mean = np.zeros(trainShape[1:], np.float32)

# loop over train nameesses
for i in range(len(trainImagesNames)):
    # print how many images are saved every 1000 images
    if i % 1000 == 0 and i > 1:
    	print ("Train data:" + str(i) + "/" + str(len(trainImagesNames)))

    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    name = trainImagesNames[i]
    img = cv2.imread(name)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # add any image pre-processing here

    # if the data order is Theano, axis orders should change
    if data_order == 'th':
        img = np.rollaxis(img, 2)

    # save the image and calculate the mean so far,  None is a list of one item
    hdf5File["train_set_x"][i, ...] = img[None]
    # the mean is calculated for training data only
    mean += img / float(len(trainLabels))

# loop over validation nameesses
for i in range(len(valImagesNames)):
    # print how many images are saved every 1000 images
    #if i % 1000 == 0 and i > 1:
    #     print ("Validation data:" + str(i) + "/" + str(len(valImagesNames)))
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    name = valImagesNames[i]
    img = cv2.imread(name)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # add any image pre-processing here

    # if the data order is Theano, axis orders should change
    if data_order == 'th':
        img = np.rollaxis(img, 2)

    # save the image, None is a list of one item
    hdf5File["val_set_x"][i, ...] = img[None]

# loop over test nameesses
for i in range(len(testImagesNames)):
    # print how many images are saved every 1000 images
    #if i % 1000 == 0 and i > 1:
    #	print ("Test data:" + str(i) + "/" + str(len(testImagesNames)))
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    name = testImagesNames[i]
    img = cv2.imread(name)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # add any image pre-processing here

    # if the data order is Theano, axis orders should change
    if data_order == 'th':
        img = np.rollaxis(img, 2)

    # save the image,  None is a list of one item
    hdf5File["test_set_x"][i, ...] = img[None]

# save the mean and close the hdf5 file
hdf5File["train_mean"][...] = mean
hdf5File.close()