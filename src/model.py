"""Main module to define, compile and train the NN model"""
import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM

from utils.generator import DataGenerator


class Mnist:
    def __init__(self) -> None:
        self.model = self.build()

    def build(self, num_classes: int = 10, name: str = "mnist") -> KM.Model:
        """Creates a classifier model object

        Args:
            num_classes (int, optional): # class labels. Defaults to 10.
            name (str, optional): name for the model object. Defaults to "mnist".

        Returns:
            KM.Model: classifier model
        """
        input_tensor = KL.Input(shape=(28, 28, 1))
        encoded = KL.Conv2D(
            4,
            3,
            strides=(2, 2),
            padding="same",
            use_bias=False,
            name=name + f"_conv_{1}",
        )(input_tensor)
        encoded = KL.Activation("relu")(KL.BatchNormalization()(encoded))
        encoded = KL.Conv2D(
            8,
            3,
            strides=(2, 2),
            padding="same",
            use_bias=False,
            name=name + f"_conv_{2}",
        )(encoded)
        encoded = KL.Activation("relu")(KL.BatchNormalization()(encoded))
        encoded = KL.GlobalAveragePooling2D()(encoded)

        # Probability vector
        prob = KL.Dense(num_classes)(encoded)

        return KM.Model(input_tensor, prob, name=name)

    def compile(self, lr: float = 1e-3) -> None:
        """method to compile the model object with optimizer, loss definitions and metrics
        Args:
            lr (float, optional): learning rate for training. defaults to 1e-3
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        metric = "accuracy"
        loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, label_smoothing=0.1
        )
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metric)

    def train(
        self,
        epochs: int = 10,
        train_batch_size: int = 32,
        lr: float = 1e-3,
        cache: bool = False,
    ) -> None:
        """method to initiate model training

        Args:
            epochs (int, optional): total number of training epochs. Defaults to 10.
            train_batch_size (int, optional): batchsize for train dataset. Defaults to 32.
            lr (float, optional): learning rate for training. defaults to 1e-3
            cache (bool, optional): whether to store the train/val data in cache. defaults to False
        """

        self.compile(lr=lr)

        train_loader = DataGenerator(
            "train", batch_size=train_batch_size, shuffle=True, cache=cache
        )
        self.model.fit(train_loader(), epochs=epochs, verbose=2)


if __name__ == "__main__":
    model = Mnist()
    model.train()
