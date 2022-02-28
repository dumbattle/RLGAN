import tensorflow as tf
import numpy as np
from Models import *
from PIL import Image
import settings
from tqdm import tqdm


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

    agent = A2CAgent(input_shape)

    discriminator = Discriminator(input_shape)
    discriminator.call(f)

    d_buf = DiscriminatorBuffer(10000, input_shape)
    d_trainer = DiscriminatorTrainer(d_buf, dataset, discriminator)

    for _ in tqdm(range(200), "Initializing Discriminator Buffer"):
        img = tf.random.uniform((1, *input_shape), 0, 1)

        for _ in range(100):
            mean = agent.actor(img)
            action = agent.sample(mean, 1)
            img = agent.update_img(img, action)
        d_buf.add(img)

    g_trainer = A2CTrainer(agent, discriminator, d_buf)
    while True:
        d_trainer.train(100)
        g_trainer.run()


if __name__ == "__main__":
    main()
