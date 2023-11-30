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
LABELS = np.array(["Genuine", "Forged"])
BATCH_SIZE = 16
EPOCH_SIZE = 10
                      
image_shape = (None, 100, 100, 3)

def cnn_train():
    print(f'\n\n{datetime.time}\n\n')
    train_ds, val_ds = loader.loader_for_cnn(batch_size=BATCH_SIZE, image_width=IMG_WIDTH,
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

def snn_train():
    SNNMODEL = model.snn_model()
    orig_train, orig_test, orig_val, forg_train, forg_test, forg_val = loader.loader_for_snn()
    #train_data = loader.generate_snn_data(orig_train, forg_train)
    #val_data = loader.generate_snn_data(orig_val, forg_val)
    # history = SNNMODEL.fit(
    #     #train_data,
    #     loader.generate_snn_batch(orig_train, forg_train, batch_size=BATCH_SIZE),
    #     epochs=24,
    #     #validation_data=val_data,
    #     validation_data=loader.generate_snn_batch(orig_val, forg_val, batch_size=BATCH_SIZE),
    #     callbacks=functions.callbacks()
    # )

    SNNMODEL.summary()




def main():
    snn_train()
    #cnn_train()

if __name__ == '__main__':
    main()
