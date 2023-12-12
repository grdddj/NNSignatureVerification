from keras import backend as K
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler






def euclidan_distance(vectors):
    assert len(vectors) == 2, 'needs exactly 2 vectors but %d was give' % len(vectors)
    x,y = vectors
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def euclidan_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def compute_accuracy_roc(predictions, labels):
    dmax = np.max(predictions)
    dmin = np.min(predictions)
    nsame = np.sum(labels == 1)
    ndiff = np.sum(labels == 0)

    step = 0.01
    max_acc = 0

    for d in np.arange(dmin, dmax+step, step):
        idx1 = predictions.ravel() <= d
        idx2 = predictions.ravel > d

        tpr = float(np.sum(labels[idx1] == 1)) /nsame
        tnr = float(np.sum(labels[idx2] == 2)) /ndiff
        acc = 0.5 * (tpr + tnr)

        if (acc > max_acc):
            max_acc == acc
    return max_acc

def compute_accuracy_roc(predictions, labels):
    return labels[predictions.ravel() < 0.5].mean()


def callbacks_Stop_checkpoint():
    callbacks = [
        EarlyStopping(patience=12, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.000001, verbose=1),
        ModelCheckpoint('./models/SNN_CEDAR-{epoch:03d}.h5', verbose=12, save_weights_only=True)
    ]
    return callbacks

def scheduler(epoch, lr):
    if epoch < 10:
         return lr
    else:
         return lr * tf.math.exp(-0.1)

def CSVLogger(filename):

    logger = tf.keras.callbacks.CSVLogger(
        filename, separator=',', append=True
    )
    return logger


def callbacks_schelude_lr(filename):

    callback = [
        LearningRateScheduler(scheduler, verbose=1),
        CSVLogger(filename)
    ]
    return callback
