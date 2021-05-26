"""data generator module"""
from typing import Tuple

import tensorflow as tf
import tensorflow_datasets as tfds


class DataGenerator:
    """Train/test dataset generator class

    Args:
        split (str, optional): dataset split to use. Defaults to "train".
        batch_size (int, optional): Defaults to 32.
        shuffle (bool, optional): whether to shuffle the dataset. Defaults to False.
        cache (bool, optional): dataset will be cached or not. Defaults to False.
    """

    def __init__(
        self,
        split: str = "train",
        batch_size: int = 32,
        shuffle: bool = False,
        cache: bool = False,
    ) -> None:
        assert split in ["train", "test", "val"]
        # Retrieve the dataset
        dataset, ds_info = tfds.load(
            "fashion_mnist",
            split="train" if split in ["train", "val"] else split,
            with_info=True,
        )

        # Extract the number of label classes
        self.num_classes = ds_info._features["label"].num_classes

        # Implement 80:20 train-val split from original "train" split
        total_size = dataset.cardinality().numpy()
        if split == "train":
            dataset = dataset.take(int(0.8 * total_size))
        elif split == "val":
            dataset = dataset.skip(int(0.8 * total_size))

        buffer_multiplier = 20 if split == "train" else 5
        if cache:
            dataset = dataset.cache()

        if shuffle:
            dataset = dataset.shuffle(
                batch_size * buffer_multiplier, reshuffle_each_iteration=True
            )

        # Per image mapping
        dataset = dataset.map(
            self.map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        dataset = dataset.batch(batch_size, drop_remainder=split == "train")

        self.dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    def map_fn(self, inputs: dict) -> Tuple[tf.Tensor, tf.Tensor]:
        """method to transform the dataset elements to model usable form

        Args:
            inputs (dict): an element from the dataset

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: input/output tensor pair
        """
        # Get the image array
        image = tf.cast(inputs["image"], tf.float32) / 255.0

        return image, tf.one_hot(inputs["label"], self.num_classes)

    def __call__(self, *args, **kwargs) -> tf.data.Dataset:
        # returns the Dataset object
        return self.dataset

    def __len__(self) -> int:
        # Get the total number of batches = epoch size
        return self.dataset.cardinality().numpy()


if __name__ == "__main__":
    train_gen = DataGenerator("train", batch_size=1, shuffle=True)
    val_gen = DataGenerator("val", batch_size=1, shuffle=False)
    print(len(train_gen))
    print(len(val_gen))
    train_loader = train_gen()
    for item in train_loader.take(1):
        print(item)
