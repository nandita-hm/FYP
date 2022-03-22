import os, json, sys
import gzip
import numpy as np
import random
import math
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tqdm import trange

from PIL import Image

# function that would read an image provided the image path, preprocess and return it back

def read_and_preprocess(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR) # reading the image
    img = cv2.resize(img, (28, 28)) # resizing it (I just like it to be powers of 2)
    img = np.array(img, dtype='float32') # convert its datatype so that it could be normalized
    img = img/255 # normalization (now every pixel is in the range of 0 and 1)
    return img

def load_mnist(img_path):
    X_train = [] # To store train images
    y_train = [] # To store train labels

    # labels -
    # 0 - Covid
    # 1 - Viral Pneumonia
    # 2 - Normal

    train_path = "Covid19-dataset/"+img_path # path containing training image samples
    for folder in os.scandir(train_path):
        for entry in os.scandir(train_path + folder.name):

            X_train.append(read_and_preprocess(train_path + folder.name + '/' + entry.name))
            
            if folder.name[0]=='C':
                y_train.append(0)
            elif folder.name[0]=='V':
                y_train.append(1)
            else:
                y_train.append(2)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    return X_train,y_train
    


def generate_dataset():

  X_train, y_train = load_mnist("train/")
  X_test, y_test = load_mnist("test/")
  print("Xtrain ")
  print(X_train.shape)
  print("ytrain ")
  print(y_train.shape)
  print("Xtest ")
  print(X_test.shape)
  print("ytest ")
  print(y_test.shape)

  # some simple normalization
  mu = np.mean(X_train.astype(np.float32), 0)
  sigma = np.std(X_train.astype(np.float32), 0)

  X_train_whole = (X_train.astype(np.float32) - mu)/(sigma+0.001)
  X_test_whole = (X_test.astype(np.float32) - mu)/(sigma+0.001)

  X_train = []
  X_test = []

  for i in range(3):
      idx = np.where(y_train == i)[0]
      X_train.append(X_train_whole[idx].tolist())
      idx = np.where(y_test == i)[0]
      X_test.append(X_test_whole[idx].tolist())

      print(len(X_train[i]))

  return X_train, y_train.tolist(), X_test, y_test.tolist()


def main():


    NUM_USER = 20

    train_output = "./data/train/my_train.json"
    test_output = "./data/test/my_test.json"


    X_train, _, X_test, _ = generate_dataset()

    X_user_train = [[] for _ in range(NUM_USER)]
    y_user_train = [[] for _ in range(NUM_USER)]

    X_user_test = [[] for _ in range(NUM_USER)]
    y_user_test = [[] for _ in range(NUM_USER)]

    # Create data structure
    train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    test_data = {'users': [], 'user_data':{}, 'num_samples':[]}

    np.random.seed(233)
    num_samples = np.random.lognormal(3, 1, NUM_USER) + 3 # 4, 1.5 for data 1
    num_samples = 300 * num_samples / sum(num_samples)  # normalize

    class_per_user = np.ones(NUM_USER) * 5
    idx_train = np.zeros(3, dtype=np.int64)
    idx_test = np.zeros(3, dtype=np.int64)
    for user in range(NUM_USER):
        props = np.random.lognormal(1, 1, int(class_per_user[user]))
        props = props / sum(props)
        for j in range(int(class_per_user[user])):
            class_id = (user + j) % 3
            train_sample_this_class = int(props[j] * num_samples[user]) + 1
            test_sample_this_class = int(props[j] * num_samples[user] / 6) + 1

            if idx_train[class_id] + train_sample_this_class  > len(X_train[class_id]):
                idx_train[class_id] = 0
            if idx_test[class_id] + test_sample_this_class  > len(X_test[class_id]):
                idx_test[class_id] = 0

            X_user_train[user] += X_train[class_id][idx_train[class_id]: (idx_train[class_id] + train_sample_this_class)]
            X_user_test[user] += X_test[class_id][idx_test[class_id]: (idx_test[class_id] + test_sample_this_class)]

            y_user_train[user] += (class_id * np.ones(train_sample_this_class)).tolist()
            y_user_test[user] += (class_id * np.ones(test_sample_this_class)).tolist()

            idx_train[class_id] += train_sample_this_class
            idx_test[class_id] += test_sample_this_class

        print('num train: ', len(X_user_train[user]), 'num test: ', len(X_user_test[user]))
        print('train labels: ', np.unique(np.asarray(y_user_train[user])), 'test labels', np.unique(np.asarray(y_user_test[user])))


    for i in range(NUM_USER):
        uname = 'f_{0:05d}'.format(i)

        combined = list(zip(X_user_train[i], y_user_train[i]))
        random.shuffle(combined)
        X_user_train[i][:], y_user_train[i][:] = zip(*combined)

        train_data['users'].append(uname)
        train_data['user_data'][uname] = {'x': X_user_train[i], 'y': y_user_train[i]}
        train_data['num_samples'].append(len(y_user_train[i]))

        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': X_user_test[i], 'y': y_user_test[i]}
        test_data['num_samples'].append(len(y_user_test[i]))


    with open(train_output, 'w') as outfile:
        json.dump(train_data, outfile)
    with open(test_output, 'w') as outfile:
        json.dump(test_data, outfile)


if __name__ == "__main__":
    main()
