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

    agent = SACAgent(input_shape)

    discriminator = Discriminator(input_shape)
    discriminator.call(f)

    d_buf = DiscriminatorBuffer(10000, input_shape)
    d_trainer = DiscriminatorTrainer(d_buf, dataset, discriminator)
    # while True:
    #     d_trainer.train_disc(agent, 10000)
    #     d_trainer.train_gen(agent, 10000)

    g_trainer = SACTrainer(agent, discriminator, data)

    @tf.function
    def _buf_init_step():
        img = tf.random.uniform((1, *input_shape), 0, 1)

        for _ in range(100):
            mean = agent.actor(img)
            action = agent.sample(mean, 1)
            img = agent.update_img(img, action)
        return img

    while True:
        for _ in tqdm(range(200), "Updating Discriminator Buffer"):
            d_buf.add(_buf_init_step())
        d_trainer.train(100, agent)
        g_trainer.run()


if __name__ == "__main__":
    main()
