"""Main module to define, compile and train the NN model"""
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM


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
        prob = KL.Dense(num_classes, activation="sigmoid")(encoded)

        return KM.Model(input_tensor, prob, name=name)


if __name__ == "__main__":
    model = Mnist()
