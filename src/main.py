import tensorflow as tf
import numpy as np
from model import *
from PIL import Image


# parameter
dataset_path = "datasets/pkmn_small_ds.npy"


def main():
    data = np.load(dataset_path)
    data = data.astype("float32")
    data /= 255
    f = data[0]
    f = tf.expand_dims(f, 0)

    discriminator = Discriminator()
    discriminator.call(f)

    gen = RLGenerator()
    a, p, r, c = gen.call(f)

    f = tf.squeeze(f, 0) * 255
    r = tf.squeeze(r, 0) * 255
    f = f.numpy().astype("uint8")
    r = r.numpy().astype("uint8")

    im = Image.fromarray(f, mode="RGBA")
    im.show()
    im = Image.fromarray(r, mode="RGBA")
    im.show()


if __name__ == "__main__":
    main()
