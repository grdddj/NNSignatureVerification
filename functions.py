import matplotlib.pyplot as plt
from PIL import Image
from keras import backend as K
import numpy as np
import tensorflow as tf
import pywt
import cv2
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler


def show_single_image(img):
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.show(block=True)


#Euclidan distance
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

# Funkce na získání více pocet tahu
def get_image_strokes(img):
    _, inverted_image = cv2.threshold(img, 0.5, 1, cv2.THRESH_BINARY_INV)
    inverted_image = np.array(inverted_image, dtype=np.uint8)
    contours, _ = cv2.findContours(inverted_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    strokes = len(contours)
    return strokes


#Local features
def image_for_local(image):
    small_image_array = []
    size = 10
    for row in range(0, image.shape, size):
        for col in range(0, image.shape, size):
            small_image = image[row:row+size, col:col+size]
            small_image_array.append(small_image)
    small_image_array = np.array(small_image, dtype="float32")
    return small_image_array

#Vlnková transformace
def wavelet_transformation(image):
    coeffs = pywt.dwt2(data=image, wavelet='bior1.3') # zkus všechny další experiment: haar, db2, bior1.3, sym, coif (https://pywavelets.readthedocs.io/en/latest/regression/wavelet.html)
    cA, (cH, cV, cD) = coeffs

    #normalizace
    cA = (cA - cA.min()) / (cA.max() - cA.min())
    cH = (cH - cH.min()) / (cH.max() - cH.min())
    #Prahování
    #_, cA = cv2.threshold(cA, 0.1, 1, cv2.THRESH_BINARY)
    #_, cH = cv2.threshold(cH, 0.1, 1, cv2.THRESH_BINARY)
    # CV a CD Nejsou potreba nic se na nich nestane nebo jsem lopata

    #Udělání jedné featury
    cA = cA.flatten()
    cH = cH.flatten()
    wavelet_features = np.concatenate((cA,cH))
    return wavelet_features

#geometric features?

# Histogram //TODO FUUUUUUCK
def calculate_histogram(image):
    _, image = cv2.threshold(image, 0.5, 1, cv2.THRESH_BINARY_INV)
    image *= 255
    image = np.array(image, dtype=np.uint8)
    # show_single_image(image)

    #hist, bins = np.histogram(image)
    hist = cv2.calcHist([image], [0], None, [256], [0,256])
    #hist = hist.squeeze()
    #hist.astype(np.int64)
    hist /= hist.sum()

    plt.plot(hist)
    plt.show(block=True)

    cdf = np.cumsum(hist)

    # Plot CDF
    plt.plot( cdf)
    plt.title('Cumulative Distribution Function (CDF)')
    plt.xlabel('Pixel Value')
    plt.ylabel('Cumulative Probability')
    plt.xlim(0, 255)
    plt.ylim(0, 1)
    plt.show(block=True)
    return cdf





# Count of pixels and representation of count of pixel upon y and x axis:
def plot_non_white_pixels(non_white_pixels_rows, non_white_pixels_columns):
    # Plot non-white pixels in rows
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(non_white_pixels_rows, range(len(non_white_pixels_rows)), color='blue')
    plt.title('Non-white Pixels in Rows')
    plt.xlabel('Count')
    plt.ylabel('Row Index')
    plt.gca().invert_yaxis()  # Invert y-axis to match image orientation

    # Plot non-white pixels in columns
    plt.subplot(1, 2, 2)
    plt.plot(range(len(non_white_pixels_columns)), non_white_pixels_columns, color='red')
    plt.title('Non-white Pixels in Columns')
    plt.xlabel('Column Index')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.show(block=True)





def count_none_white_pixels(image, count_axis=False):
    pixel_count = np.sum(image == 0)
    if count_axis:
        non_white_pixels_rows = np.sum(image == 0, axis=1)
        non_white_pixels_columns = np.sum(image == 0, axis=0)
        plot_non_white_pixels(non_white_pixels_rows, non_white_pixels_columns)
        return pixel_count, non_white_pixels_rows, non_white_pixels_columns
    return pixel_count



# Center of mass, Normalized area ,Aspect Ratio, Tri surface feature,six fold surface feature and Transition feature

# morphological features





