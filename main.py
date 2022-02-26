import tensorflow as tf
import numpy as np
from Models import *
from PIL import Image
import settings
from tqdm import  tqdm

def main():
    data = np.load(settings.dataset_path)
    data = data.astype("float32")
    data /= 255

    dataset = tf.data.Dataset\
        .from_tensor_slices(data)\
        .shuffle(data.shape[0])\
        .batch(settings.Discriminator.Training.batch_size)

    f = data[0]
    input_shape = f.shape
    f = tf.expand_dims(f, 0)

    discriminator = Discriminator(input_shape)
    discriminator.call(f)

    gen = RLGen(input_shape)

    d_buf = DiscriminatorBuffer(10000, input_shape)
    d_trainer = DiscriminatorTrainer(d_buf, dataset, discriminator)

    for i in tqdm(range(100), "Initializing Discriminator Buffer"):
        img = tf.random.uniform((1, *input_shape), 0, 1)
        for i in range(1):
            mean, _ = gen.call(f)
            action = gen.sample(mean, 1)
            img = img + action / 255 / gen.magnitude
        d_buf.add(img)

    g_buf = RLGenBuffer(input_shape, 100)
    g_trainer = RLGenTrainer(gen, discriminator, input_shape, g_buf, d_buf)
    while True:
        d_trainer.train(100)
        g_trainer.run()

    # f = tf.squeeze(f, 0) * 255
    # r = tf.squeeze(r, 0) * 255
    # f = f.numpy().astype("uint8")
    # r = r.numpy().astype("uint8")
    #
    # im = Image.fromarray(f, mode="RGBA")
    # im.show()
    # im = Image.fromarray(r, mode="RGBA")
    # im.show()


if __name__ == "__main__":
    main()
