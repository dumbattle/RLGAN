from Models import *
import settings
from tqdm import tqdm
import os
import tensorflow as tf
from utils import generate_noisy_input, generate_blotched_input, display_images, generate_blotched_demo
import matplotlib.pyplot as plt

def pad(x):
    return tf.pad(x, [[0, 0], [0, 1], [11, 12], [0, 0]])


def load_A2CD_1():
    dir = "saves/A2CD-1"

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


def load_TD3_C10():
    dir = "saves/TD3-C10-B"

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    data = x_train
    data = data.astype("float32")
    data /= 255

    dataset = tf.data.Dataset\
        .from_tensor_slices(data)\
        .shuffle(data.shape[0])\
        .batch(settings.Discriminator.Training.batch_size)

    for f in dataset.take(1):
        input_shape = f.shape[1:]
    discriminator = Discriminator(input_shape)

    agent = TD3Agent(input_shape)
    d_buf = DiscriminatorBuffer(10000, input_shape)

    d_trainer = DiscriminatorTrainer(d_buf, dataset, discriminator, dir, data)
    d_trainer.load()

    g_trainer = TD3Trainer(agent, discriminator, data, dir)
    g_trainer.load()
    return discriminator, agent, d_trainer, g_trainer, d_buf, dir


def load_TD3_1():
    def augment(x):
        x = tf.image.random_flip_left_right(x)
        x2 = x[:, :, :, :-1]
        x2 = tf.image.random_hue(x2, 0.5)
        x3 = x[:, :, :, -1:]
        return tf.concat((x2, x3), -1)
    dir = "saves/TD3-B-3"
    data = np.load(settings.dataset_path)

    data = data.astype("float32")
    data /= 255

    data = pad(data).numpy()

    dataset = tf.data.Dataset\
        .from_tensor_slices(data)\
        .shuffle(data.shape[0])\
        .batch(settings.Discriminator.Training.batch_size, True)\
        .map(lambda x: augment(x),
                num_parallel_calls=tf.data.AUTOTUNE)

    for f in dataset.take(1):
        input_shape = f.shape[1:]
    discriminator = Discriminator(input_shape)

    agent = TD3Agent(input_shape)
    d_buf = DiscriminatorBuffer(10000, input_shape)

    d_trainer = DiscriminatorTrainer(d_buf, dataset, discriminator, dir, data)
    d_trainer.load()

    g_trainer = TD3Trainer(agent, d_trainer.disc, data, dir)
    g_trainer.load()
    return discriminator, agent, d_trainer, g_trainer, d_buf, dir


def load_TD3_2():
    dir = "saves/TD3-2"

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

    d_trainer = DiscriminatorTrainer(d_buf, dataset, discriminator,dir)
    d_trainer.load()

    g_trainer = TD3Trainer(agent, discriminator, data, dir, delta_score=True)
    g_trainer.load()
    return discriminator, agent, d_trainer, g_trainer, d_buf, dir


def load_TD3_SN():
    dir = "saves/TD3-SN"

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
    discriminator = DiscriminatorSN(input_shape)

    agent = TD3Agent(input_shape, True)
    d_buf = DiscriminatorBuffer(10000, input_shape)

    d_trainer = DiscriminatorTrainer(d_buf, dataset, discriminator,dir)
    d_trainer.load()

    g_trainer = TD3Trainer(agent, discriminator, data, dir)
    g_trainer.load()
    return discriminator, agent, d_trainer, g_trainer, d_buf, dir


def load_TD3_3():
    dir = "saves/TD3-3"

    data = np.load(settings.dataset_path)

    data = data.astype("float32")
    data /= 255.0

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

    d_trainer = DiscriminatorTrainer(d_buf, dataset, discriminator, dir)
    d_trainer.load()

    g_trainer = TD3Trainer(agent, discriminator, data, dir)
    g_trainer.load()
    return discriminator, agent, d_trainer, g_trainer, d_buf, dir


def run(agent, d_buf, g_trainer, d_trainer, save_dir):
    @tf.function
    def _buf_init_step():
        return agent.generate(generate_blotched_input(g_trainer.real, 1))

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
            # for _ in tqdm(range(64), "Updating Discriminator Buffer"):
            #     d_buf.add(_buf_init_step())
            d_trainer.train(1000, agent)
            current_phase = 2

        elif current_phase == 2:
            g_trainer.run(epoch)
            current_phase = 0

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        np.savez(f'{save_dir}/state', current_phase=current_phase, epoch=epoch)


def demo(agent, data):
    while True:
        img = generate_blotched_input(data, 16)
        agent.generate(img=img, steps=500, count=9, display=True)


def main():
    # discriminator, agent, d_trainer, g_trainer, d_buf, save_dir = load_TD3_SN()
    # discriminator, agent, d_trainer, g_trainer, d_buf, save_dir = load_TD3_C10()
    discriminator, agent, d_trainer, g_trainer, d_buf, save_dir = load_TD3_1()
    # discriminator, agent, d_trainer, g_trainer, d_buf, save_dir = load_TD3_2()
    # discriminator, agent, d_trainer, g_trainer, d_buf, save_dir = load_TD3_3()
    # discriminator, agent, d_trainer, g_trainer, d_buf, save_dir = load_A2C_1()
    # discriminator, agent, d_trainer, g_trainer, d_buf, save_dir = load_A2C_2()
    # discriminator, agent, d_trainer, g_trainer, d_buf, save_dir = load_A2CD_1()

    # print(discriminator(generate_noisy_input(g_trainer.real, 256)))

    run(agent, d_buf, g_trainer, d_trainer, save_dir)
    # demo(agent, g_trainer.real)
    # data = np.load(settings.dataset_path)
    #
    # data = data.astype("float32")
    # data /= 255
    #
    # data = pad(data).numpy()
    # a = generate_blotched_demo(data)
    # while True:
    # #     a = generate_blotched_input(data, 9)
    #     display_images([a])
    #     # display_images(np.array([a]))
    #     pass
    # a = data[75]
    # c = tf.random.uniform(a.shape, 0, 1)
    # b = a * .95 + c * .05
    #
    # plt.figure()
    # for i in range(3):
    #     plt.subplot(1, 4, i + 1)
    #     plt.title(["Score: 0.99", "Score: 0.01", "Score: 0.00"][i])
    #     plt.imshow([a, b, c][i])
    #     plt.axis('off')
    # plt.show()
    # while True:
    #     pass
    #     # display_images(np.array([a, b, c]))

    # sample = data[np.random.choice(len(data), 16)]
    # sample2 = display_images(sample, False)
    #
    # im = Image.fromarray(sample2)
    # im.save(f"saves/Data demo.png")
    # while True:
    #     display_images(sample, False)
    #     pass


if __name__ == "__main__":
    main()
