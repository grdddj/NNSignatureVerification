"""
@author: Petr Čírtek
"""
# Not used imports
import keras
import PIL
from keras import backend
from keras import layers, models
from keras.backend import relu, sigmoid
# True imports
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, MaxPooling2D, \
    BatchNormalization, ZeroPadding2D, Input, Lambda
from tensorflow.keras.regularizers import L1L2, L1, L2

import functions
from functions import euclidan_distance, euclidan_dist_output_shape
from keras.optimizers import SGD, RMSprop, Adadelta




def cnn_model(image_shape=(None,100 , 100, 1)):
    # konfigurace vrstev
    num_conv_filters = 32        # pocet conv. filtru
    max_pool_size = (2, 2)       # velikost maxpool filtru
    conv_kernel_size = (3, 3)    # velikost conv. filtru
    imag_shape = image_shape     # vlastnosti obrazku
    dropout_prob = 0.25          # pravdepodobnost odstraneni neuronu

    # Predspracovani funkce
    rescale = Sequential([
        layers.Rescaling(1./255)
    ])
    threshold = Sequential([
        layers.ThresholdedReLU(theta=0.6)
    ])
    rotate = Sequential([
        layers.RandomRotation(factor=(-0.05, 0.05))
    ])
    translate = Sequential([
        layers.RandomTranslation(height_factor=(-0.1, 0.1), width_factor=(-0.1,0.1))
    ])
    zoom = Sequential([
        layers.RandomZoom(height_factor=(-0.1, 0.1), width_factor=(-0.1,0.1))
    ])
    # normalize = Sequential([
    #     layers.Normalization()
    # ])


    model = Sequential()  # Typ modelu

    #Preprocessed Vrstva
    model.add(rescale)
    model.add(threshold)
    #model.add(rotate)
    #model.add(translate)
    #model.add(zoom)

    # 1. vrstva
    model.add(Conv2D(filters=num_conv_filters, kernel_size=(conv_kernel_size), input_shape=imag_shape,
                     activation='relu', kernel_regularizer=L1L2(l1=0.1e-4, l2=0.1e-5),
                     #bias_regularizer=L1(l1=0.01), activity_regularizer=L2(l2=0.01)
                     ))
    model.add(MaxPooling2D(pool_size=max_pool_size))
    model.add(Dropout(dropout_prob))

    # 2. vrstva
    model.add(Conv2D(filters=num_conv_filters * 2, kernel_size=(conv_kernel_size), input_shape=imag_shape,
                     activation='relu', kernel_regularizer=L1L2(l1=0.1e-4, l2=0.1e-5),
                     #bias_regularizer=L1(l1=0.01), activity_regularizer=L2(l2=0.01)
                     ))
    model.add(MaxPooling2D(pool_size=max_pool_size))

    # 3.
    model.add(Conv2D(filters=num_conv_filters * 4, kernel_size=(conv_kernel_size), input_shape=imag_shape,
                     activation='relu', kernel_regularizer=L1L2(l1=0.1e-4, l2=0.1e-5),
                     #bias_regularizer=L1(l1=0.01), activity_regularizer=L2(l2=0.01)
                     ))
    model.add(MaxPooling2D(pool_size=max_pool_size))
    model.add(Dropout(dropout_prob))


    # Plne propojena vrstva
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))

    # odstraneni neuronu proti overfittingu
    model.add(Dropout(dropout_prob*2))

    # Vyhodnoceni
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def snn_base_cnn_model(image_shape=(100 , 100, 1)):
    num_conv_filters = 32  # pocet conv. filtru
    max_pool_size = (2, 2)  # velikost maxpool filtru
    conv_kernel_size = (3, 3)  # velikost conv. filtru
    imag_shape = image_shape  # vlastnosti obrazku
    dropout_prob = 0.25  # pravdepodobnost odstraneni neuronu
    model = Sequential()


    model.add(Conv2D(filters=num_conv_filters, kernel_size=(conv_kernel_size[0]*2, conv_kernel_size[1]*2),
                        input_shape=imag_shape, activation='relu', data_format='channels_last'
                     ))
    model.add(BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9))
    model.add(MaxPool2D(pool_size=max_pool_size))


    model.add(Conv2D(filters=num_conv_filters*2, kernel_size=(conv_kernel_size), input_shape=imag_shape,
                        activation='relu', data_format='channels_last'
                     ))
    model.add(BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9))
    model.add(MaxPool2D(pool_size=max_pool_size))
    model.add(Dropout(dropout_prob))


    model.add(Conv2D(filters=num_conv_filters*3, kernel_size=(conv_kernel_size), input_shape=imag_shape,
                        activation='relu', data_format='channels_last'
                     ))
    model.add(MaxPool2D(pool_size=max_pool_size))

    model.add(Flatten())
    model.add(Dense(512, kernel_regularizer=L2(l2=0.1e-5), activation='relu'))
    model.add(Dropout(dropout_prob*2))

    model.add(Dense(128, activation='relu'))

    # model.add(Conv2D(filters=96, kernel_size=(11, 11), activation='relu', name='conv1_1', strides=4, input_shape=imag_shape), padding='same')
    # model.add(BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9))
    # model.add(MaxPooling2D((3,3), strides=(2, 2)))
    # model.add(ZeroPadding2D((2, 2)))

    # model.add(Conv2D(filters=256, kernel_size=(5, 5), activation='relu', name='conv2_1', strides=1))
    # model.add(BatchNormalization(epsilon=1e-06, axis=1, momentum=0.9))
    # model.add(MaxPooling2D((3,3), strides=(2, 2)))
    # model.add(Dropout(0.3))
    # model.add(ZeroPadding2D((1, 1)))
    #
    # model.add(Conv2D(filters=384, kernel_size=(3, 3), activation='relu', name='conv3_1', strides=1))
    # model.add(ZeroPadding2D((1, 1)))
    #
    # model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', name='conv3_2', strides=1))
    # model.add(MaxPooling2D((3,3), strides=(2, 2)))
    # model.add(Dropout(0.3))
    # model.add(Flatten(name='flatten'))
    # model.add(Dense(1024, W_regularizer=L2(l2=0.0005), activation='relu'))
    # model.add(Dropout(0.5))
    #
    # model.add(Dense(128, W_regularizer=L2(l2=0.0005), activation='relu'))
    print(model.summary())
    return model


def snn_model(image_shape=(100, 100, 1)):
    base_network = snn_base_cnn_model(image_shape)
    image1 = Input(shape=(image_shape))
    print(f'\nshape of im1 is {image1.shape}' )
    image2 = Input(shape=(image_shape))
    print(f'\nshape of im2 is {image2.shape}')
    preprocessed_image1 = base_network(image1)
    print(preprocessed_image1.shape)
    preprocessed_image2 = base_network(image2)
    print(preprocessed_image2.shape)

    distance = Lambda(euclidan_distance, output_shape=euclidan_dist_output_shape)([preprocessed_image1, preprocessed_image2])
    model = Model(inputs=[image1, image2], outputs=distance)
    rms = RMSprop(learning_rate=1e-4, rho=0.9, epsilon=1e-08)
    model.compile(loss=functions.contrastive_loss, optimizer=rms)

    return model



# def debug():
#     Model = cnn_model()
#     Model.build((None,200,200,1))
#     Model.summary()
