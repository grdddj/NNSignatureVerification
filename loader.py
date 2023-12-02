"""
@author: Petr Čírtek
"""
import glob
import itertools
import os
import imghdr
import time
import random
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.utils import shuffle

import PIL
from PIL import Image
import tensorflow as tf
import tensorflow_datasets as tfds

IMAGE_TYPES = ['jpeg', 'png', 'bmp', 'png']
DATASET_NUM_CLASSES = {
    'cedar': 55,
}


# DATASET_SIGNATURES_PER_PERSON = {
#     'cedar_org': 24,
#     'cedar_forg': 24,
# }


# Vizualizace dat:
def plot_images(image_array, image_array_label=[], num_column=5, title='Images in dataset'):
    fig, axes = plt.subplots(1, num_column)
    fig.suptitle('This is a somewhat long figure title', fontsize=16)
    axes = axes.flatten()
    index = 0
    for img, ax in zip(image_array, axes):
        ax.imshow(img, cmap='Greys_r')
        if image_array_label != []:
            ax.set_title(image_array_label[index])
        index += 1
        ax.axis('off')
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()



def loader_for_cnn(data_dir=None, image_width=200, image_height=200, batch_size=16):
    # mam tam osobni s dlouhym i takze musim solo zadat ze data
    if (not data_dir):
        data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
    print(data_dir)

    # Hozeni dat do datasetu a roztrideni na VAL a TRAIN datasety
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset='training',
        seed=123,
        image_size=(image_width, image_height),
        batch_size=batch_size,
        color_mode='grayscale',
        shuffle=True,
        # crop_to_aspect_ratio=True,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset='validation',
        seed=123,
        image_size=(image_width, image_height),
        batch_size=batch_size,
        color_mode='grayscale',
        shuffle=True,
        # crop_to_aspect_ratio=True,
    )

    class_names = train_ds.class_names
    print(train_ds)
    print(class_names)
    class_names = val_ds.class_names
    print(val_ds)
    print(class_names)

    print('??????????????????__________________________________________?????????????????????')

    iterator = iter(train_ds)
    sample_of_train_dataset, _ = next(iterator)

    plot_images(sample_of_train_dataset[:5])
    return train_ds, val_ds


# SNN LOADER
def create_snn_data(data_dir, dataset='cedar'):
    num_classes = DATASET_NUM_CLASSES[dataset]
    images = glob.glob(data_dir + '/*.png')
    num_of_signatures = int(len(images) / num_classes) # this only works with Cedar
    print(images)
    # labels = []
    persons = []
    index = 0
    for person in range(num_classes):
        signatures = []
        for signature in range(num_of_signatures):
            signatures.append(images[index])
            index += 1
        persons.append(signatures)
    train_data, test_data, val_data = persons[:43], persons[43:45], persons[45:]
    print(train_data)
    print(test_data)
    print(val_data)
    return train_data, test_data, val_data


def convert_to_image(image_path, img_w=150, img_h=150):
    img = Image.open(image_path)
    img = img.resize((img_w, img_h))
    img = img.convert('L')
    img = img.point(lambda p: 255 if p > 200 else 0)  # Thresholding
    img = img.convert('1')  # udela to to co chci??
    img = np.array(img, dtype='float32')
    img /= 255
    img = img[..., np.newaxis]
    return img


def convert_pairs_to_image_pairs(pair_array, labels, img_w=150, img_h=150, output_size=0):
    image_pair_array = []
    new_labels = []
    index = 0

    if output_size == 0:
        output_size = len(pair_array)
        for pair in pair_array:
            if (index == output_size):
                break
            image1 = convert_to_image(pair[0])
            image2 = convert_to_image(pair[1])
            image_pair_array.append((image1, image2))
            new_labels.append(labels[index])
            index += 1
        return image_pair_array, new_labels

    rng = np.random.default_rng();
    indieces = rng.choice(len(pair_array), size=output_size, replace=False, shuffle=True)
    for i in indieces:
        image1 = convert_to_image(pair_array[i][0])
        image2 = convert_to_image(pair_array[i][1])
        image_pair_array.append((image1, image2))
        new_labels.append(labels[i])
        index += 1
    del pair_array, labels
    return image_pair_array, new_labels


def make_pairs(orig_data, forg_data):
    orig_pairs, forg_pairs = [], []
    # if output_size == 0:
    for orig, forg in zip(orig_data, forg_data):
        orig_pairs.extend(list(itertools.combinations(orig, 2)))
        for i in range(len(forg)):
            forg_pairs.extend(list(itertools.product(orig[i:i + 1], random.sample(forg, 12))))
    data_pairs = orig_pairs + forg_pairs
    orig_pair_labels = [1] * len(orig_pairs)
    orig_forg_labels = [0] * len(forg_pairs)

    label_pairs = orig_pair_labels + orig_forg_labels
    del orig_data, forg_data
    print(f'len of orig_pars: {len(orig_pairs)}')
    print(f'len of forg_pairs {len(forg_pairs)}')
    del orig_pairs, forg_pairs
    print('final results')
    print(len(data_pairs))
    print(len(label_pairs))
    print('testing accuracy')
    for i in range(5):
        num = np.random.randint(0, len(data_pairs))
        testing_pair = data_pairs[num]
        print(len(testing_pair))
        print(f'{testing_pair[0]} , {testing_pair[1]} = {label_pairs[num]}')
    return data_pairs, label_pairs


def visualize_snn_sample_signature_for_signer(orig_data, forg_data, image_width=200, image_height=200):
    k = np.random.randint(len(orig_data))
    orig_data_signature = random.sample(orig_data[k], 2)
    forg_data_signature = random.sample(forg_data[k], 1)
    print(orig_data_signature[0])
    print(orig_data_signature[1])
    print(forg_data_signature[0])
    orig_im1 = cv2.imread(orig_data_signature[0], 0)
    orig_im1 = cv2.resize(orig_im1, (image_width, image_height))
    orig_im2 = cv2.imread(orig_data_signature[1], 0)
    orig_im2 = cv2.resize(orig_im2, (image_width, image_height))
    forg_im = cv2.imread(forg_data_signature[0], 0)
    forg_im = cv2.resize(forg_im, (image_width, image_height))
    img_array_to_show = [orig_im1, orig_im2, forg_im]
    img_array_label = ['genuine', 'genuine', 'forgery']
    plot_images(img_array_to_show, img_array_label, num_column=len(img_array_to_show))


def show_pair(pairs, labels, title='Image pairs', columns=2, rows=1):
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 5, rows * 2))
    fig.suptitle(title)
    if rows == 1:
        axes[0].imshow(pairs[0][0], cmap='gray')
        axes[0].set_title(labels[0][0])
        axes[0].axis('off')
        axes[1].imshow(pairs[0][1], cmap='gray')
        axes[1].set_title(labels[0][1])
        axes[1].axis('off')
    else:
        for row in range(rows):
            img_pair = pairs[row]
            label = labels[row]
            for column in range(columns):
                axes[row, column].imshow(img_pair[column], cmap='gray')
                axes[row, column].set_title(label[column])
                axes[row, column].axis('off')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.show(block=True)


def visualize_snn_pair_sample(pair_array, label_array, title='Pair sample', numer_of_samples=5):
    pairs = []
    label = []
    for i in range(numer_of_samples):
        k = np.random.randint(0, len(pair_array))
        img1 = pair_array[k][0]
        img2 = pair_array[k][1]
        pairs.append([img1, img2])
        label.append(['Geunine', 'Genuine' if label_array[k] == 1 else 'Forgery'])

    show_pair(pairs, label, title=title, columns=2, rows=numer_of_samples)


def loader_for_snn(data_dir=None, train_size=6000, val_size=1500, test_size=500, image_width=200, image_height=200,
                   dataset='cedar'):
    path_to_orig = 'data/genuine'
    path_to_forg = 'data/forgery'

    start_time = time.time()
    # Tohle nahraje do listu, kde se obr8ykz nachazi
    '''Tohle vraci array typu clovek[podpis1[], podpis2[]], clovek2[podpis1[], podpis2[]] tak na to nezapominej'''
    orig_train, orig_test, orig_val = create_snn_data(path_to_orig, dataset=dataset)
    forg_train, forg_test, forg_val = create_snn_data(path_to_forg, dataset=dataset)

    # visualize_snn_sample_signature(orig_train, forg_train)

    print(f'ORIG TEST: {len(orig_test)}')
    print(f'FORG TEST: {len(forg_test)}')

    '''Tohle vraci array typu clovek[obrazek1[], obrazek2[]] a tak dale jak predtim tak na to kurva nezapominej'''
    # orig_val = convert_to_images(orig_val, True) #redundantni boolean smaz pak
    # forg_val = convert_to_images(forg_val, False)
    print('___________________TESTOVACI SADA___________________')
    test_pairs, test_labels = make_pairs(orig_test, forg_test)
    print('___________________VYTVORENY PARY__________________')
    test_pairs, test_labels = convert_pairs_to_image_pairs(test_pairs, test_labels, img_w=image_width,
                                                           img_h=image_height, output_size=test_size)
    print('___________________TRENOVACI SADA___________________')
    train_pairs, train_labels = make_pairs(orig_train, forg_train)
    print('___________________VYTVORENY PARY__________________')
    train_pairs, train_labels = convert_pairs_to_image_pairs(train_pairs, train_labels, img_w=image_width,
                                                             img_h=image_height, output_size=train_size)
    print('___________________Validacni SADA___________________')
    val_pairs, val_labels = make_pairs(orig_val, forg_val)
    print('___________________VYTVORENY PARY__________________')
    val_pairs, val_labels = convert_pairs_to_image_pairs(val_pairs, val_labels, img_w=image_width, img_h=image_height,
                                                         output_size=val_size)

    print('_____________________HOTOVO__________________________________')
    end_time = time.time()
    print(f'trvalo to : {end_time - start_time}')
    print(f'Trenovaci sada: {len(train_pairs)} , labels: {len(train_labels)} and shape = {train_pairs[0][0].shape}')
    print(f'Testovaci sada: {len(test_pairs)}, labels: {len(test_labels)} and shape = {train_pairs[0][0].shape}')
    print(f'Validacni sada: {len(val_pairs)} , labels: {len(val_labels)} and shape = {train_pairs[0][0].shape}')

    visualize_snn_pair_sample(train_pairs, train_labels, title='Train pairs', numer_of_samples=5)
    visualize_snn_pair_sample(test_pairs, test_labels, title='Test pairs', numer_of_samples=1)
    visualize_snn_pair_sample(val_pairs, val_labels, title='Validation pairs', numer_of_samples=2)

    # orig_set = create_data(path_to_orig, dataset=dataset)
    # orig_train, orig_test, orig_val = orig_set[0], orig_set[1], orig_set[2]
    # del orig_set
    # forg_set = create_data(path_to_forg, dataset=dataset)
    # forg_train, forg_test, forg_val = forg_set[0], forg_set[1], forg_set[2]
    # del forg_set

    # visualize_snn_sample_signature(orig_train, forg_train)

    print('ending the story')
    #
    # batch = generate_snn_batch(orig_train, forg_train, batch_size=batch_size)
    # print(batch)
    train_pairs, train_labels = np.array(train_pairs), np.array(train_labels, dtype=np.float32)
    test_pairs, test_labels = np.array(test_pairs), np.array(test_labels, dtype=np.float32)
    val_pairs, val_labels = np.array(val_pairs), np.array(val_labels, dtype=np.float32)

    return train_pairs, train_labels, test_pairs, test_labels, val_pairs, val_labels
