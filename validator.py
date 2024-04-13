import glob

import keras
import tensorflow as tf
from tensorflow.keras.models import load_model

import loader


def test_model(datadir="fromServer"):
    models = glob.glob(datadir + "/*.h5")
    for i in range(len(models)):
        print(f"{i}: {models[i]}")
    index = int(input("Choose model: "))
    modelPath = models[index]
    isSnn = int(input("SNN (0) or CNN (1): "))
    numTestSamples = int(input("Number of test samples: "))
    batch_size = int(input("Batch size: "))
    width = int(input("Image width: "))
    height = int(input("Image height: "))

    model = load_model(modelPath)

    if isSnn == 1:
        numTestSamples = int(numTestSamples / 2)
        data, labels = loader.loader_for_cnn(
            data_dir="tester",
            image_width=width,
            image_height=height,
            augmented=False,
            size=numTestSamples,
            dataset="test",
        )
    else:
        print(numTestSamples)
        pairs, labels = loader.loader_for_snn(
            data_dir="tester",
            image_width=width,
            image_height=height,
            size=numTestSamples,
            dataset="test",
        )
    isEval = 200
    while isEval != -1:
        isEval = int(input("Evaluate(0) or predict(1) or end(-1): "))
        if isEval == 0:
            if isSnn == 0:
                result = model.evaluate(
                    x=([pairs[:, 0, :, :], pairs[:, 1, :, :]]),
                    y=labels,
                    batch_size=batch_size,
                )
            else:
                result = model.evaluate(x=data, y=labels, batch_size=batch_size)
            print(f"test loss and acc = {result}")
        elif isEval == 1:
            numOfPred = int(input("Number of predictions: "))
            if isSnn == 0:
                new_pairs = pairs[:numOfPred]
                prediction = model.predict(
                    [new_pairs[:, 0, :, :], new_pairs[:, 1, :, :]]
                )
            else:
                new_data = data[:numOfPred]
                prediction = model.predict(new_data)
            print(f"prediction shape: {prediction.shape}")
            for i in range(len(prediction)):
                print(f"predictions: {prediction[i]} for lable: {labels[i]}")
