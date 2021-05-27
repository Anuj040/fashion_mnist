"""Main module to define, compile and train the NN model"""
import glob
import os
import sys

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM
from tensorflow.keras.callbacks import ModelCheckpoint

sys.path.append("./")
# pylint: disable = wrong-import-position
from src.utils.generator import DataGenerator


class Mnist:
    """Mnist fashion model class"""

    def __init__(self) -> None:
        self.model = self.build()
        self.class_names = [
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ]

    def load_model(self, path: str) -> None:
        """method to load the trained model

        Args:
            path (str): path to models directory

        Returns:
            KM.Model: trained model object
        """
        models = glob.glob(os.path.join(path, "*.h5"))
        models = sorted(models)

        self.model = KM.load_model(models[-1])

    def build(self, num_classes: int = 10, name: str = "mnist") -> KM.Model:
        """Creates a classifier model object

        Args:
            num_classes (int, optional): # class labels. Defaults to 10.
            name (str, optional): name for the model object. Defaults to "mnist".

        Returns:
            KM.Model: classifier model
        """
        input_tensor = KL.Input(shape=(28, 28, 1))

        encoded = KL.Flatten()(input_tensor)
        encoded = KL.Dense(128, activation="relu", name=name + "den_1")(encoded)

        # Probability vector
        prob = KL.Dense(num_classes, name=name + "den_2")(encoded)

        return KM.Model(input_tensor, prob, name=name)

    def compile(self, l_rate: float = 1e-3) -> None:
        """method to compile the model object with optimizer, loss definitions and metrics
        Args:
            l_rate (float, optional): learning rate for training. defaults to 1e-3
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=l_rate)
        metric = "accuracy"
        loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, label_smoothing=0.1
        )
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metric)

    # pylint: disable = too-many-arguments
    def train(
        self,
        epochs: int = 10,
        train_batch_size: int = 32,
        val_batch_size: int = 32,
        l_rate: float = 1e-3,
        cache: bool = False,
    ) -> dict:
        """method to initiate model training

        Args:
            epochs (int, optional): total number of training epochs. Defaults to 10.
            train_batch_size (int, optional): batchsize for train dataset. Defaults to 32.
            val_batch_size (int, optional): batchsize for val dataset. Defaults to 32.
            l_rate (float, optional): learning rate for training. defaults to 1e-3
            cache (bool, optional): whether to store the train/val data in cache. defaults to False

        Returns:
            dict: training history [loss, accuracy]
        """

        self.compile(l_rate=l_rate)

        train_loader = DataGenerator(
            "train", batch_size=train_batch_size, shuffle=True, cache=cache
        )
        val_loader = DataGenerator(
            "val", batch_size=val_batch_size, shuffle=False, cache=cache
        )
        history = self.model.fit(
            train_loader(),
            epochs=epochs,
            validation_data=val_loader(),
            verbose=2,
            workers=8,
            callbacks=[
                ModelCheckpoint(
                    "save_model/model_{val_accuracy:.4f}.h5",
                    "val_accuracy",
                    save_best_only=True,
                    mode="max",
                )
            ],
        )

        return history.history

    def eval(self) -> str:
        """model evaluation method

        Returns:
            str: model evaluation metrics
        """
        test_loader = DataGenerator("test", batch_size=32, shuffle=False)
        history = self.model.evaluate(test_loader(), verbose=0, workers=8)
        return (
            "=" * 60
            + f"\nTest set evaluation: loss = {history[0]:.4f} and accuracy = {history[1]:.4f}.\n"
            + "=" * 60
        )

    def infer(self, image: np.ndarray) -> str:
        """model inference method

        Args:
            image (np.ndarray): image array

        Returns:
            str: prediction for the image
        """

        history = self.model(tf.expand_dims(image, axis=0), training=False)
        label = tf.argmax(history, axis=-1).numpy()

        category = self.class_names[label[0]]
        return (
            "=" * (34 + len(category))
            + f"\nThis image belongs to '{category}' category.\n"
            + "=" * (34 + len(category))
        )


if __name__ == "__main__":
    model = Mnist()

    # model.train(cache=True)

    model.load_model("save_model")
    # print(model.eval())
    img = tf.keras.preprocessing.image.load_img(
        "test_samples/test.png", color_mode="grayscale"
    )
    img = tf.keras.preprocessing.image.img_to_array(img, dtype=float) / 255.0
    print(model.infer(img))
