"""
@author: Petr Čírtek
"""

import glob
import itertools
import os
import random
import time

import matplotlib
import sklearn.utils
from scipy import ndimage

matplotlib.use("TkAgg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

IMAGE_TYPES = ["jpeg", "png", "bmp", "png"]
DATASET_NUM_CLASSES = {
    "cedar": 55,
    "test": 1,
}


# DATASET_SIGNATURES_PER_PERSON = {
#     'cedar_org': 24,
#     'cedar_forg': 24,
# }


# Vizualizace dat:
def plot_images(
    image_array, image_array_label=[], num_column=5, title="Images in dataset"
):
    fig, axes = plt.subplots(1, num_column, figsize=(20, 20))
    fig.suptitle(title, fontsize=16)
    axes = axes.flatten()
    index = 0
    for img, ax in zip(image_array, axes):
        ax.imshow(img, cmap="Greys_r")
        if image_array_label != []:
            ax.set_title(image_array_label[index])
            index += 1
        # ax.axis('off')

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show(block=True)


# Loader z cesty
def create_for_tr_ts_val_data(data_dir, dataset="cedar"):
    num_classes = DATASET_NUM_CLASSES[dataset]
    images = glob.glob(data_dir + "/*.png")
    num_of_signatures = int(len(images) / num_classes)  # this only works with Cedar
    # print(images)
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
    # print(train_data)
    # print(test_data)
    # print(val_data)
    return train_data, test_data, val_data


def create_data(data_dir, dataset="cedar"):
    num_classes = DATASET_NUM_CLASSES[dataset]
    images = glob.glob(data_dir + "/*.png")
    num_of_signatures = int(len(images) / num_classes)  # this only works with Cedar
    if num_of_signatures == 0:
        num_of_signatures = 1
    # print(images)
    persons = []
    index = 0
    for person in range(num_classes):
        signatures = []
        for signature in range(num_of_signatures):
            signatures.append(images[index])
            index += 1
        persons.append(signatures)
    return persons


# menic na obrazky
def convert_to_image(image_path, img_w=150, img_h=150):
    img = Image.open(image_path)
    img = img.resize((img_w, img_h))
    img = img.convert("L")
    img = img.point(lambda p: 255 if p > 210 else 0)  # Thresholding
    img = img.convert("1")  # udela to to co chci?? ANO
    img = np.array(img, dtype="float32")
    img = img[..., np.newaxis]
    return img


# Augmentations:
def rand_rotate(imag):
    img = imag.copy()
    w = img.shape[1]
    h = img.shape[0]
    if np.random.randint(0, 2) == 0:
        if np.random.randint(0, 2) == 0:
            angle = 10
        else:
            angle = -10
    else:
        if np.random.randint(0, 2) == 0:
            angle = 20
        else:
            angle = -20

    matrix = cv2.getRotationMatrix2D(
        (w / 2, h / 2), angle, 1.0
    )  # center cx, cy = w/2 h/2
    img = cv2.warpAffine(
        img, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(1, 1, 1)
    )
    return img


def rand_translation(imag):
    img = imag.copy()
    shift = 10.0
    width = img.shape[1]
    height = img.shape[0]
    direction = np.random.randint(0, 4)

    if direction == 0:  # UP
        matrix = np.float32([[1, 0, 0], [0, 1, -shift]])
    if direction == 1:  # DOWN
        matrix = np.float32([[1, 0, 0], [0, 1, shift]])
    if direction == 2:  # RIGHT
        matrix = np.float32([[1, 0, -shift], [0, 1, 0]])
    if direction == 3:  # LEFT
        matrix = np.float32([[1, 0, shift], [0, 1, 0]])
    img = cv2.warpAffine(
        img,
        matrix,
        (width, height),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(1, 1, 1),
    )
    return img


def rand_zoom(imag):
    img = imag.copy()
    zoom = float(np.random.randint(8, 12)) / 10
    cy, cx = [i / 2 for i in img.shape[:-1]]
    matrix = cv2.getRotationMatrix2D((cx, cy), 0, zoom)
    img = cv2.warpAffine(
        img,
        matrix,
        img.shape[1::-1],
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(1, 1, 1),
    )
    return img


def rand_shear(imag):
    img = imag.copy()
    axe = np.random.randint(0, 2)
    width = img.shape[1]
    height = img.shape[0]
    if axe == 0:
        matrix = np.float32([[1, 0.5, 0], [0, 1, 0], [0, 0, 1]])
    if axe == 1:
        matrix = np.float32([[1, 0, 0], [0.5, 1, 0], [0, 0, 1]])

    img = cv2.warpPerspective(
        img,
        matrix,
        (int(width * 1.2), int(height * 1.2)),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(1, 1, 1),
    )

    return img


def rand_noise(imag):
    img = imag.copy()
    gaussian_noise = np.random.normal(0, 0.3, img.shape)
    img += gaussian_noise
    img = np.clip(img, 0, 1)
    return img


# Agmentator
def augment_image(
    img,
    rotate=True,
    shear=True,
    zoom=True,
    shift=True,
    gaussian_noice=True,
    save_path=None,
):
    augmented_images = []

    # Random rotation (plus minus 10 degrees)
    rotated_image = rand_rotate(img)
    rotated_image = rotated_image[..., np.newaxis]
    augmented_images.append(rotated_image)

    # Random shear
    sheared_image = rand_shear(img)
    sheared_image = cv2.resize(sheared_image, (img.shape[0], img.shape[1]))
    sheared_image = sheared_image[..., np.newaxis]
    augmented_images.append(sheared_image)

    # Random zoom
    resized_image = rand_zoom(img)
    resized_image = resized_image[..., np.newaxis]
    augmented_images.append(resized_image)

    # Random shift
    shifted_image = rand_translation(img)
    shifted_image = shifted_image[..., np.newaxis]
    augmented_images.append(shifted_image)

    # Gaussian noise
    noisy_image = rand_noise(img)
    augmented_images.append(noisy_image)

    # Save augmented images if specified
    if save_path:
        for i, augmented_image in enumerate(augmented_images):
            save_image_path = save_path.format(i)
            ndimage.imsave(save_image_path, augmented_image.squeeze())

    #  plot_images(augmented_images, ['rotated', 'sheared', 'zoomed', 'shifted', 'gaussian noise'], title="Augmentation")
    return augmented_images


def convert_array_to_image_labels(
    image_path_array,
    image_width=150,
    image_height=150,
    augmented=False,
    genuine=False,
    size=None,
):
    labels = []
    image_array = []
    for person in image_path_array:
        for img in person:
            img = convert_to_image(img)
            image_array.append(img)
            labels.append(1 if genuine else 0)
            if augmented:
                augmented_images = augment_image(img)
                image_array.extend(augmented_images)
                # augmented_images.insert(0,img)
                # augmented_labels = ['Původní', 'Otočený', 'Smýknutý', 'Přiblížený/Oddálený', 'Posunutý', 'Ss šumem']
                # plot_images(augmented_images,  num_column=6,
                #            title='Upravené obrázky') #THIS ONLY FOR SHOWING PURPOSES
                if genuine:
                    labels.extend([1 for i in range(len(augmented_images))])
                else:
                    labels.extend([0 for i in range(len(augmented_images))])
    # image_array = np.array(image_array)
    # labels = np.array(labels, dtype=np.float32)
    if size:
        image_sized_array = []
        label_sized_array = []
        rng = np.random.default_rng()
        indieces = rng.choice(len(image_array), size=size, replace=False, shuffle=True)
        for i in indieces:
            image = image_array[i]
            image_sized_array.append(image)
            label_sized_array.append(labels[i])
        return image_sized_array, label_sized_array
    return image_array, labels


def combine_orig_forg(orig_data, forg_data, orig_labels, forg_labels, shuffle=True):
    data = orig_data + forg_data
    labels = orig_labels + forg_labels
    if shuffle:
        data, labels = sklearn.utils.shuffle(data, labels, random_state=42)

    return data, labels


# CNN Loader
def loader_for_cnn(
    data_dir="data",
    image_width=150,
    image_height=150,
    dataset="cedar",
    augmented=False,
    size=None,
):

    path_to_orig = data_dir + "/genuine"
    path_to_forg = data_dir + "/forgery"

    start_time = time.time()

    # THIS IS CURRENT :]
    orig_data = create_data(path_to_orig, dataset=dataset)
    forg_data = create_data(path_to_forg, dataset=dataset)
    print(f"ORIG DATA: {len(orig_data)}")
    print(f"FORG DATA: {len(forg_data)}")
    orig_data, orig_labels = convert_array_to_image_labels(
        orig_data, genuine=True, augmented=augmented, size=size
    )
    forg_data, forg_labels = convert_array_to_image_labels(
        forg_data, genuine=False, augmented=augmented, size=size
    )  # Moznost augmentace
    print(f"ORIG DATA: {len(orig_data)}")
    print(f"FORG DATA: {len(forg_data)}")
    data, labels = combine_orig_forg(orig_data, forg_data, orig_labels, forg_labels)
    print(f"Dataset: {len(data)} and labels: {len(labels)}")

    data, labels = np.array(data), np.array(labels, dtype=np.float32)

    end_time = time.time()
    print(end_time - start_time)
    return data, labels


def tensor_loader_for_cnn(
    data_dir=None, image_width=200, image_height=200, batch_size=16
):
    # mam tam osobni s dlouhym i takze musim solo zadat ze data
    if not data_dir:
        data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
    print(data_dir)

    # Hozeni dat do datasetu a roztrideni na VAL a TRAIN datasety
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(image_width, image_height),
        batch_size=batch_size,
        color_mode="grayscale",
        shuffle=True,
        # crop_to_aspect_ratio=True,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(image_width, image_height),
        batch_size=batch_size,
        color_mode="grayscale",
        shuffle=True,
        # crop_to_aspect_ratio=True,
    )

    class_names = train_ds.class_names
    print(train_ds)
    print(class_names)
    class_names = val_ds.class_names
    print(val_ds)
    print(class_names)

    print(
        "??????????????????__________________________________________?????????????????????"
    )

    # iterator = iter(train_ds)
    # sample_of_train_dataset, _ = next(iterator)

    # plot_images(sample_of_train_dataset[:5])
    return train_ds, val_ds


# SNN LOADER


def convert_pairs_to_image_pairs(
    pair_array, labels, img_w=150, img_h=150, output_size=0
):
    image_pair_array = []
    new_labels = []
    index = 0

    if output_size == 0:
        output_size = len(pair_array)
        for pair in pair_array:
            if index == output_size:
                break
            image1 = convert_to_image(pair[0])
            image2 = convert_to_image(pair[1])
            image_pair_array.append((image1, image2))
            new_labels.append(labels[index])
            index += 1
        return image_pair_array, new_labels

    rng = np.random.default_rng()
    indieces = rng.choice(
        len(pair_array), size=output_size, replace=False, shuffle=True
    )
    for i in indieces:
        image1 = convert_to_image(pair_array[i][0])
        image2 = convert_to_image(pair_array[i][1])
        image_pair_array.append((image1, image2))
        new_labels.append(labels[i])
    del pair_array, labels
    return image_pair_array, new_labels


def make_pairs(orig_data, forg_data):
    orig_pairs, forg_pairs = [], []
    orig_feature, forg_feature = [], []

    # if output_size == 0:
    for orig, forg in zip(orig_data, forg_data):
        orig_pairs.extend(list(itertools.combinations(orig, 2)))
        for i in range(len(forg)):
            forg_pairs.extend(
                list(itertools.product(orig[i : i + 1], random.sample(forg, 12)))
            )
    data_pairs = orig_pairs + forg_pairs
    orig_pair_labels = [1] * len(orig_pairs)
    orig_forg_labels = [0] * len(forg_pairs)

    label_pairs = orig_pair_labels + orig_forg_labels
    del orig_data, forg_data
    print(f"len of orig_pars: {len(orig_pairs)}")
    print(f"len of forg_pairs {len(forg_pairs)}")
    del orig_pairs, forg_pairs
    print("final results")
    print(len(data_pairs))
    print(len(label_pairs))
    print("testing accuracy")
    # for i in range(5):
    #     num = np.random.randint(0, len(data_pairs))
    #     testing_pair = data_pairs[num]
    #     print(len(testing_pair))
    #     print(f'{testing_pair[0]} , {testing_pair[1]} = {label_pairs[num]}')
    return data_pairs, label_pairs


def visualize_snn_sample_signature_for_signer(
    orig_data, forg_data, image_width=200, image_height=200
):
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
    img_array_label = ["genuine", "genuine", "forgery"]
    plot_images(img_array_to_show, img_array_label, num_column=len(img_array_to_show))


def show_pair(pairs, labels, title="Image pairs", columns=2, rows=1):
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 5, rows * 2))
    fig.suptitle(title)
    if rows == 1:
        axes[0].imshow(pairs[0][0], cmap="gray")
        axes[0].set_title(labels[0][0])
        axes[0].axis("off")
        axes[1].imshow(pairs[0][1], cmap="gray")
        axes[1].set_title(labels[0][1])
        axes[1].axis("off")
    else:
        for row in range(rows):
            img_pair = pairs[row]
            label = labels[row]
            for column in range(columns):
                axes[row, column].imshow(img_pair[column], cmap="gray")
                axes[row, column].set_title(label[column])
                axes[row, column].axis("off")
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.show(block=True)


def visualize_snn_pair_sample(
    pair_array, label_array, title="Pair sample", numer_of_samples=5
):
    pairs = []
    label = []
    for i in range(numer_of_samples):
        k = np.random.randint(0, len(pair_array))
        img1 = pair_array[k][0]
        img2 = pair_array[k][1]
        pairs.append([img1, img2])
        label.append(["Geunine", "Genuine" if label_array[k] == 1 else "Forgery"])

    # show_pair(pairs, label, title=title, columns=2, rows=numer_of_samples)


def loader_for_snn(
    data_dir="data",
    train_size=6000,
    val_size=1500,
    test_size=500,
    image_width=200,
    image_height=200,
    dataset="cedar",
    size=4000,
):
    path_to_orig = data_dir + "/genuine"
    path_to_forg = data_dir + "/forgery"

    print(path_to_orig)
    print(path_to_forg)

    start_time = time.time()

    orig_data = create_data(path_to_orig, dataset=dataset)
    forg_data = create_data(path_to_forg, dataset=dataset)
    print(f"ORIG : {len(orig_data)}")
    print(f"FORG : {len(forg_data)}")

    print("___________________VYTVARENI PARU__________________")
    data_pairs, data_labels = make_pairs(orig_data, forg_data)
    print("___________________Nacteni Obrazku__________________")
    data_pairs, data_labels = convert_pairs_to_image_pairs(
        data_pairs, data_labels, img_w=image_width, img_h=image_height, output_size=size
    )
    print("_____________________HOTOVO__________________________________")
    end_time = time.time()
    print(f"trvalo to : {end_time - start_time}")
    print(
        f"Data: {len(data_pairs)} , labels: {len(data_labels)} and shape = {data_pairs[0][0].shape}"
    )
    # visualize_snn_pair_sample(data_pairs, data_labels, title='Data pairs', numer_of_samples=5)
    data_pairs, data_labels = np.array(data_pairs), np.array(
        data_labels, dtype=np.float32
    )
    return data_pairs, data_labels
