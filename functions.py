import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pywt
import shap
import tensorflow as tf
from keras import backend as K
from keras import utils
from keras.callbacks import (
    EarlyStopping,
    LearningRateScheduler,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from PIL import Image

# TODO TODO TODO Natrenovat pro kazdou mnozinu, a udelat experimenty, spojit mnoziny a natrenovat na te vizulizovat pri predikci
# TODO TODO Predelat main aby to davalo vetsi smysl, pridat installation do README a requirements a dopsat to konecne kurva
# TODO pokud bude cas vytvorit jeste jeden model ktery bude pracovat jen s priznaky ^^


def show_single_image(img):
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.show(block=True)


# Euclidan distance
def euclidan_distance(vectors):
    assert len(vectors) == 2, "needs exactly 2 vectors but %d was give" % len(vectors)
    x, y = vectors
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def euclidan_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(
        y_true * K.square(y_pred)
        + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0))
    )


def compute_accuracy_roc(predictions, labels):
    dmax = np.max(predictions)
    dmin = np.min(predictions)
    nsame = np.sum(labels == 1)
    ndiff = np.sum(labels == 0)

    step = 0.01
    max_acc = 0

    for d in np.arange(dmin, dmax + step, step):
        idx1 = predictions.ravel() <= d
        idx2 = predictions.ravel > d

        tpr = float(np.sum(labels[idx1] == 1)) / nsame
        tnr = float(np.sum(labels[idx2] == 2)) / ndiff
        acc = 0.5 * (tpr + tnr)

        if acc > max_acc:
            max_acc == acc
    return max_acc


def compute_accuracy_roc(predictions, labels):
    return labels[predictions.ravel() < 0.5].mean()


def callbacks_Stop_checkpoint():
    callbacks = [
        EarlyStopping(patience=12, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.000001, verbose=1),
        ModelCheckpoint(
            "./models/SNN_CEDAR-{epoch:03d}.h5", verbose=12, save_weights_only=True
        ),
    ]
    return callbacks


def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


def CSVLogger(filename):

    logger = tf.keras.callbacks.CSVLogger(filename, separator=",", append=True)
    return logger


def callbacks_schelude_lr(filename):

    callback = [LearningRateScheduler(scheduler, verbose=1), CSVLogger(filename)]
    return callback


# Funkce na získání více pocet tahu
def get_image_strokes(img):
    _, inverted_image = cv2.threshold(img, 0.5, 1, cv2.THRESH_BINARY_INV)
    inverted_image = np.array(inverted_image, dtype=np.uint8)
    contours, _ = cv2.findContours(
        inverted_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    strokes = len(contours)
    return strokes


# Local features
def image_for_local(image):
    small_image_array = []
    size = 10
    for row in range(0, image.shape, size):
        for col in range(0, image.shape, size):
            small_image = image[row : row + size, col : col + size]
            small_image_array.append(small_image)
    small_image_array = np.array(small_image, dtype="float32")
    return small_image_array


# Vlnková transformace
def wavelet_transformation(image):
    coeffs = pywt.dwt2(
        data=image, wavelet="bior1.3"
    )  # zkus všechny další experiment: haar, db2, bior1.3, sym, coif (https://pywavelets.readthedocs.io/en/latest/regression/wavelet.html)
    cA, (cH, cV, cD) = coeffs

    # normalizace
    cA = (cA - cA.min()) / (cA.max() - cA.min())
    cH = (cH - cH.min()) / (cH.max() - cH.min())
    # Prahování
    # _, cA = cv2.threshold(cA, 0.1, 1, cv2.THRESH_BINARY)
    # _, cH = cv2.threshold(cH, 0.1, 1, cv2.THRESH_BINARY)
    # CV a CD Nejsou potreba nic se na nich nestane nebo jsem lopata

    # Udělání jedné featury
    cA = cA.flatten()
    cH = cH.flatten()
    wavelet_features = np.concatenate((cA, cH))
    wavelet_features = wavelet_features.flatten()
    return wavelet_features


# Count of pixels and representation of count of pixel upon y and x axis (histogram):
def plot_non_white_pixels(non_white_pixels_rows, non_white_pixels_columns):
    # Plot non-white pixels in rows
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(non_white_pixels_rows, range(len(non_white_pixels_rows)), color="blue")
    plt.title("Non-white Pixels in Rows")
    plt.xlabel("Count")
    plt.ylabel("Row Index")
    plt.gca().invert_yaxis()  # Invert y-axis to match image orientation

    # Plot non-white pixels in columns
    plt.subplot(1, 2, 2)
    plt.plot(
        range(len(non_white_pixels_columns)), non_white_pixels_columns, color="red"
    )
    plt.title("Non-white Pixels in Columns")
    plt.xlabel("Column Index")
    plt.ylabel("Count")

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
# https://www.researchgate.net/publication/258650160_Handwritten_Signature_Verification_using_Neural_Network
def calculate_center_of_mass(image):
    width = image.shape[1]
    height = image.shape[0]
    half_width = width // 2
    first_half = image[:half_width, :]
    second_half = image[half_width:, :]

    M1 = cv2.moments(first_half)
    M2 = cv2.moments(second_half)

    center_of_mass_first_half_x = M1["m10"] / M1["m00"] if M1["m00"] != 0 else 0
    center_of_mass_first_half_y = M1["m01"] / M1["m00"] if M1["m00"] != 0 else 0
    center_of_mass_second_half_x = M2["m10"] / M2["m00"] if M2["m00"] != 0 else 0
    center_of_mass_second_half_y = M2["m01"] / M2["m00"] if M2["m00"] != 0 else 0

    # normalization
    center_of_mass_first_half_normalized = np.array(
        [center_of_mass_first_half_x / half_width, center_of_mass_first_half_y / height]
    )
    center_of_mass_second_half_normalized = np.array(
        [
            center_of_mass_second_half_x / half_width,
            center_of_mass_second_half_y / height,
        ]
    )

    return center_of_mass_first_half_normalized, center_of_mass_second_half_normalized


def calculate_normalized_shape(image):
    image = image[:, :, 0]
    img_area = np.sum(image == 0)
    pixel_indieces = np.where(image == 0)
    rows, cols = pixel_indieces
    bounding_box_area = (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)
    normalized_shape = img_area / bounding_box_area
    return normalized_shape


def calculate_aspect_ratio(image):
    image = image[:, :, 0]
    pixel_indieces = np.where(image == 0)
    rows, cols = pixel_indieces
    height = max(rows) - min(rows) + 1
    width = max(cols) - min(cols) + 1
    aspect_ratio = width / height

    return aspect_ratio


def calculate_tri_surface_area(image):
    tri_width = image.shape[1] // 3
    parts = [image[:, i * tri_width : (i + 1) * tri_width] for i in range(3)]
    areas = [calculate_normalized_shape(part) for part in parts]
    return areas


def six_fold_surface(image):
    image = image[:, :, 0]
    part_width = image.shape[1] // 3
    parts = [image[:, i * part_width : (i + 1) * part_width] for i in range(3)]
    all_features = []
    for part in parts:
        features = []
        pixel_indieces = np.where(part == 0)
        rows, cols = pixel_indieces
        boundingbox_width = [min(cols), max(cols)]  # width
        boundingbox_height = [min(rows), max(rows)]  # height
        bounding_box = [
            boundingbox_width[1] - boundingbox_width[0],
            boundingbox_height[1] - boundingbox_height[0],
        ]
        features.append(bounding_box)
        print(f"Width of bound {boundingbox_width}")
        print(f"Height of bound {boundingbox_height}")
        Mx = cv2.moments(
            part[
                boundingbox_height[0] : boundingbox_height[1],
                boundingbox_width[0] : boundingbox_width[1],
            ]
        )
        center_of_mass_x = Mx["m10"] / Mx["m00"] if Mx["m00"] != 0 else 0
        center_of_mass_y = Mx["m01"] / Mx["m00"] if Mx["m00"] != 0 else 0
        print(f"Mass = {center_of_mass_x} x {center_of_mass_y}")
        area_above_center = np.sum(
            part[
                boundingbox_height[0] : (boundingbox_height[0] + int(center_of_mass_y)),
                boundingbox_width[0] : boundingbox_width[1],
            ]
            == 0
        )
        area_bellow_center = np.sum(
            part[
                (boundingbox_height[0] + int(center_of_mass_y)) : boundingbox_height[1],
                boundingbox_width[0] : boundingbox_width[1],
            ]
            == 0
        )
        print(f"Bellow {area_bellow_center} and Above {area_above_center}")
        # show_single_image(part[boundingbox_height[0]:(boundingbox_height[0] + int( center_of_mass_y)), boundingbox_width[0]:boundingbox_width[1]])
        # show_single_image(part[(boundingbox_height[0] + int(center_of_mass_y)):boundingbox_height[1], boundingbox_width[0]:boundingbox_width[1]])
        center_of_mass_x = center_of_mass_x / (
            boundingbox_width[1] - boundingbox_width[0]
        )  # normalized
        center_of_mass_y = center_of_mass_y / (
            boundingbox_height[1] - boundingbox_height[0]
        )
        features.append([center_of_mass_x, center_of_mass_y])
        features.append([area_bellow_center, area_above_center])
        all_features.append(features)

    return all_features


def save_and_display_gradcam(image, heatmap, cam_path="heatmap.jpg", alpha=0.4):
    heatmap = np.unit8(255 * heatmap)
    jet = mpl.colormaps("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((image.shape[1], image.shape[0]))
    jet_heatmap = utils.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + image
    superimposed_img = utils.array_to_img(superimposed_img)
    superimposed_img.save(cam_path)
    show_single_image(superimposed_img)


def visualize_with_shap(image, model):
    shap.initjs()
    masker = shap.maskers.Image("inapaint_telea", image.shape)
    explainer = shap.Explainer(
        model, masker, [0, 1]
    )  # TODO add names accordingly to predictions
    shap_values = explainer(image)
    shap.image_plot(shap_values)


def add_features(data, isPair=True, type="strokes"):
    feature = []

    if type == "strokes":
        if isPair:
            for pair in data:
                stroke1 = get_image_strokes(pair[0])
                stroke2 = get_image_strokes(pair[1])
                feature.append([stroke1, stroke2])
    elif type == "wavelet":
        if isPair:
            for pair in data:
                wavelet1 = wavelet_transformation(pair[0])
                wavelet2 = wavelet_transformation(pair[1])
                feature.append([wavelet1, wavelet2])

    print(len(feature))
    return np.array(feature)
