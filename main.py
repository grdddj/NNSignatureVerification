from __future__ import  absolute_import, division, print_function, unicode_literals
# IMPORTING ALLES :))
import datetime
import sys
import numpy as np
import pickle
import os
import matplotlib

import functions

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


import cv2
import time
import itertools
import random

from sklearn.utils import shuffle


import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from keras.models import Sequential
from keras.optimizers import Adam
from tensorflow.keras.models import load_model
import glob
import logging


import loader
import model

IMG_HEIGHT = 150
IMG_WIDTH = 150
CHANNELS = 1
LABELS = np.array(["Forged", "Genuine"])
BATCH_SIZE = 32
EPOCH_SIZE = 10
                      
image_shape = (None, 100, 100, 3)

def cnn_train():
    train_ds, val_ds = loader.tensor_loader_for_cnn(batch_size=BATCH_SIZE, image_width=IMG_WIDTH,
                                                    image_height=IMG_HEIGHT)

    CNNMODEL = model.cnn_model()
    hist = CNNMODEL.fit(train_ds, batch_size=BATCH_SIZE, epochs=EPOCH_SIZE,
                        # steps_per_epoch=len(train_ds)/BATCH_SIZE,
                        validation_data=val_ds,
                        shuffle=True,
                        )

    # MODEL.build(image_shape)
    CNNMODEL.summary()

    print(f'\n\n{datetime.time}\n\n')

    fig = plt.figure(figsize=(7, 7))
    plt.plot(hist.history['loss'], color='teal', label='loss')
    plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
    fig.suptitle('Loss', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()

    fig = plt.figure(figsize=(7, 7))
    plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
    plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
    fig.suptitle('Accuracy', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()

    pre = Precision()
    re = Recall()
    acc = BinaryAccuracy()

    CNNMODEL.save(os.path.join('models', 'cnnSignatureVerificator.h5'))

    print('\n\n\n\nAAAAAALLL DONE')

def cnn_train_augmented():
    data, labels = loader.loader_for_cnn(image_width=IMG_WIDTH, image_height=IMG_HEIGHT)
    CNNMODEL = model.cnn_model(image_shape=(IMG_WIDTH,IMG_HEIGHT, CHANNELS))
    hist = CNNMODEL.fit(x=data,
                        y=labels,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCH_SIZE,
                        # steps_per_epoch=len(train_ds)/BATCH_SIZE,
                        validation_split=0.2,
                        shuffle=True,
                        callbacks=functions.callbacks_schelude_lr()
                        )

    # MODEL.build(image_shape)
    CNNMODEL.summary()

    print(f'\n\n{datetime.time}\n\n')

    fig = plt.figure(figsize=(7, 7))
    plt.plot(hist.history['loss'], color='teal', label='loss')
    plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
    fig.suptitle('Loss', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()

    fig = plt.figure(figsize=(7, 7))
    plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
    plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
    fig.suptitle('Accuracy', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()

    CNNMODEL.save(os.path.join('models', 'cnnSignatureVerificatorAugmented.h5'))

    print('\n\n\n\nAAAAAALLL DONE')

def snn_train():
    SNNMODEL = model.snn_model(image_shape=(IMG_WIDTH, IMG_HEIGHT, CHANNELS))
    #train_pairs, train_labels, test_pairs, test_labels, val_pairs, val_labels = loader.loader_for_snn(image_width=IMG_WIDTH, image_height=IMG_HEIGHT, test_size=100, train_size=4000, val_size=100)
    data_pairs, data_labels = loader.loader_for_snn(image_width=IMG_WIDTH, image_height=IMG_HEIGHT, size=2000)

    hist = SNNMODEL.fit(
        x=([data_pairs[:, 0, :,:], data_pairs[:,1,:,:]]),
        y=data_labels,
        #steps_per_epoch= int(len(train_pairs)/BATCH_SIZE),
        batch_size=BATCH_SIZE,
        epochs=EPOCH_SIZE,
        shuffle=True,
        validation_split=0.2,
        callbacks=functions.callbacks_schelude_lr()
        #validation_data=([val_pairs[:,0,:,:], val_pairs[:,1,:,:]], val_labels),
        #validation_steps=int(len(val_pairs)/BATCH_SIZE),
        #callbacks=functions.callbacks()
    )

    SNNMODEL.save(os.path.join('models', 'SnnSignatureVerificatorFinal.h5'))

    SNNMODEL.summary()

    fig = plt.figure(figsize=(7, 7))
    plt.plot(hist.history['loss'], color='teal', label='loss')
    plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
    fig.suptitle('Loss', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()

    fig = plt.figure(figsize=(7, 7))
    plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
    plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
    fig.suptitle('Accuracy', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()

def continue_on_cnn():
    pass

def continue_on_snn(model):
    loaded_model = load_model(model)
    data_pairs, data_labels = loader.loader_for_snn(image_width=IMG_WIDTH, image_height=IMG_HEIGHT, size=2500)
    hist = loaded_model.fit(
        x=([data_pairs[:, 0, :,:], data_pairs[:,1,:,:]]),
        y=data_labels,
        #steps_per_epoch= int(len(train_pairs)/BATCH_SIZE),
        batch_size=BATCH_SIZE,
        epochs=EPOCH_SIZE,
        shuffle=True,
        validation_split=0.2,
        callbacks=functions.callbacks_schelude_lr()
    )

    loaded_model.save(os.path.join('models', 'SnnSignatureVerificatorLoaded.h5'))

    loaded_model.summary()

    fig = plt.figure(figsize=(7, 7))
    plt.plot(hist.history['loss'], color='teal', label='loss')
    plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
    fig.suptitle('Loss', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()

    fig = plt.figure(figsize=(7, 7))
    plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
    plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
    fig.suptitle('Accuracy', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()


def main():
    ans = int(input('Do you wanna activate CNN(0) CNN AUGMENTEd(1) or SNN(2) continue training CNN(3) or SNN(4):  '))
    if ans == 0:cnn_train()
    elif ans == 1: cnn_train_augmented()
    elif ans == 2:snn_train()
    elif ans == 3: continue_on_cnn()
    elif ans == 4: continue_on_snn('models/SnnSignatureVerificatorFinal.h5')
    else:
        print("nothings gonna happen :)")
        return



if __name__ == '__main__':
    main()
