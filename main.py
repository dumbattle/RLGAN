from Models import *
import settings
from tqdm import tqdm
import os


def pad(x):
    return tf.pad(x, [[0, 0], [0, 1], [11, 12], [0, 0]])


def load_A2CD_1():
    dir = "saves/A2CD-3"

    data = np.load(settings.dataset_path)

    data = data.astype("float32")
    data /= 255

    data = pad(data).numpy()

    dataset = tf.data.Dataset\
        .from_tensor_slices(data)\
        .shuffle(data.shape[0])\
        .batch(settings.Discriminator.Training.batch_size)

    for f in dataset.take(1):
        input_shape = f.shape[1:]
    discriminator = Discriminator(input_shape)

    agent = A2CDAgent(input_shape)
    d_buf = DiscriminatorBuffer(10000, input_shape)

    d_trainer = DiscriminatorTrainer(d_buf, dataset, discriminator, dir)
    d_trainer.load()

    g_trainer = A2CDTrainer(agent, discriminator, data, dir)
    g_trainer.load()
    return discriminator, agent, d_trainer, g_trainer, d_buf, dir


def load_A2C_1():
    data = np.load(settings.dataset_path)

    data = data.astype("float32")
    data /= 255

    data = pad(data).numpy()

    dataset = tf.data.Dataset\
        .from_tensor_slices(data)\
        .shuffle(data.shape[0])\
        .batch(settings.Discriminator.Training.batch_size)

    for f in dataset.take(1):
        input_shape = f.shape[1:]
    discriminator = Discriminator(input_shape)

    agent = A2CAgent(input_shape, 3)
    d_buf = DiscriminatorBuffer(10000, input_shape)

    d_trainer = DiscriminatorTrainer(d_buf, dataset, discriminator, "saves/A2C-1")
    d_trainer.load()

    g_trainer = A2CTrainer(agent, discriminator, data, "saves/A2C-1")
    g_trainer.load()
    return discriminator, agent, d_trainer, g_trainer, d_buf, "saves/A2C-1"


def load_A2C_2():
    dir = "saves/A2C-2"
    data = np.load(settings.dataset_path)

    data = data.astype("float32")
    data /= 255

    data = pad(data).numpy()

    dataset = tf.data.Dataset\
        .from_tensor_slices(data)\
        .shuffle(data.shape[0])\
        .batch(settings.Discriminator.Training.batch_size)
    input_shape = None
    for f in dataset.take(1):
        input_shape = f.shape[1:]
    discriminator = Discriminator(input_shape)

    agent = A2CAgent(input_shape, 50, .99)
    d_buf = DiscriminatorBuffer(10000, input_shape)

    d_trainer = DiscriminatorTrainer(d_buf, dataset, discriminator, dir)
    d_trainer.load()

    g_trainer = A2CTrainer(agent, discriminator, data, dir)
    g_trainer.load()
    return discriminator, agent, d_trainer, g_trainer, d_buf, dir


def load_TD3_1():
    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # data = x_train
    # data = np.reshape(data, (-1, 28, 28, 1))

    data = np.load(settings.dataset_path)

    data = data.astype("float32")
    data /= 255

    # data = tf.pad(data, [[0, 0], [2, 2], [2, 2], [0, 0]]).numpy()
    data = pad(data).numpy()

    dataset = tf.data.Dataset\
        .from_tensor_slices(data)\
        .shuffle(data.shape[0])\
        .batch(settings.Discriminator.Training.batch_size)

    for f in dataset.take(1):
        input_shape = f.shape[1:]
    discriminator = Discriminator(input_shape)

    agent = TD3Agent(input_shape)
    d_buf = DiscriminatorBuffer(10000, input_shape)

    d_trainer = DiscriminatorTrainer(d_buf, dataset, discriminator, "saves/TD3-1")
    d_trainer.load()

    g_trainer = TD3Trainer(agent, discriminator, data, "saves/TD3-1")
    g_trainer.load()
    return discriminator, agent, d_trainer, g_trainer, d_buf, "saves/TD3-1"


def run(agent, d_buf, g_trainer, d_trainer, save_dir):
    @tf.function
    def _buf_init_step():
        return agent.generate()

    epoch = 0
    current_phase = 0

    if os.path.exists(f'{save_dir}/state.npz'):
        with np.load(f'{save_dir}/state.npz') as f:
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
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        np.savez(f'{save_dir}/state', current_phase=current_phase, epoch=epoch)


def demo(agent):
    while True:
        agent.generate(steps=500, count=9, display=True)


def main():
    discriminator, agent, d_trainer, g_trainer, d_buf, save_dir = load_TD3_1()
    # discriminator, agent, d_trainer, g_trainer, d_buf, save_dir = load_A2C_1()
    # discriminator, agent, d_trainer, g_trainer, d_buf, save_dir = load_A2C_2()
    # discriminator, agent, d_trainer, g_trainer, d_buf, save_dir = load_A2CD_1()

    run(agent, d_buf, g_trainer, d_trainer, save_dir)
    # demo(agent)


if __name__ == "__main__":
    main()
