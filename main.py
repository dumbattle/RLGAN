from Models import *
import settings
from tqdm import tqdm
import os


def main():
    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # data = x_train
    # data = np.reshape(data, (-1, 28, 28, 1))

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

    agent = TD3Agent(input_shape)
    d_buf = DiscriminatorBuffer(10000, input_shape)

    d_trainer = DiscriminatorTrainer(d_buf, dataset, discriminator)
    d_trainer.load()

    g_trainer = TD3Trainer(agent, discriminator, data)
    g_trainer.load()

    @tf.function
    def _buf_init_step():
        img = tf.random.uniform((1, *input_shape), 0, 1)

        for _ in range(100):
            mean = agent.actor(img, training=False)
            action = agent.sample(mean, 1)
            img = agent.update_img(img, action)
        return img

    epoch = 0
    current_phase = 0

    if os.path.exists('saves/state.npz'):
        with np.load('saves/state.npz') as f:
            epoch = f['epoch']
            current_phase = f['current_phase']

    # train loop
    while True:
        if current_phase == 0:
            epoch += 1
            current_phase = 1

        elif current_phase == 1:
            for _ in tqdm(range(200), "Updating Discriminator Buffer"):
                d_buf.add(_buf_init_step())
            d_trainer.train(1000, agent)
            current_phase = 2

        elif current_phase == 2:
            g_trainer.run(epoch)
            current_phase = 0
        if not os.path.exists('saves'):
            os.mkdir('saves')
        np.savez('saves/state', current_phase=current_phase, epoch=epoch)


if __name__ == "__main__":
    main()
