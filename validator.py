import glob

from tensorflow.keras.models import load_model

import loader


def test_model(datadir: str = "fromServer") -> None:
    models = glob.glob(datadir + "/*.h5")
    for i in range(len(models)):
        print(f"{i}: {models[i]}")
    index = int(input("Choose model: "))
    model_path = models[index]
    is_snn = int(input("SNN (0) or CNN (1): "))
    num_test_samples = int(input("Number of test samples: "))
    batch_size = int(input("Batch size: "))
    width = int(input("Image width: "))
    height = int(input("Image height: "))

    model = load_model(model_path)

    if is_snn == 1:
        num_test_samples = int(num_test_samples / 2)
        data, labels = loader.loader_for_cnn(
            data_dir="tester",
            image_width=width,
            image_height=height,
            augmented=False,
            size=num_test_samples,
            dataset="test",
        )
    else:
        print(num_test_samples)
        pairs, labels = loader.loader_for_snn(
            data_dir="tester",
            image_width=width,
            image_height=height,
            size=num_test_samples,
            dataset="test",
        )
    isEval = 200
    while isEval != -1:
        isEval = int(input("Evaluate(0) or predict(1) or end(-1): "))
        if isEval == 0:
            if is_snn == 0:
                result = model.evaluate(
                    x=([pairs[:, 0, :, :], pairs[:, 1, :, :]]),
                    y=labels,
                    batch_size=batch_size,
                )
            else:
                result = model.evaluate(x=data, y=labels, batch_size=batch_size)
            print(f"test loss and acc = {result}")
        elif isEval == 1:
            num_of_pred = int(input("Number of predictions: "))
            if is_snn == 0:
                new_pairs = pairs[:num_of_pred]
                prediction = model.predict(
                    [new_pairs[:, 0, :, :], new_pairs[:, 1, :, :]]
                )
            else:
                new_data = data[:num_of_pred]
                prediction = model.predict(new_data)
            print(f"prediction shape: {prediction.shape}")
            for i in range(len(prediction)):
                print(f"predictions: {prediction[i]} for lable: {labels[i]}")
