from Models import *
import settings
from tqdm import tqdm
import os

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

    agent = TD3Agent(input_shape)
    g_trainer = TD3Trainer(agent, discriminator, data)

    d_buf = DiscriminatorBuffer(10000, input_shape)
    d_trainer = DiscriminatorTrainer(d_buf, dataset, discriminator)
    # while True:
    #     d_trainer.train_disc(agent, 10000)
    #     d_trainer.train_gen(agent, 10000)

    @tf.function
    def _buf_init_step():
        img = tf.random.uniform((1, *input_shape), 0, 1)

        for _ in range(100):
            mean = agent.actor(img)
            action = agent.sample(mean, 1)
            img = agent.update_img(img, action)
        return img

    epoch = 0
    current_phase = 0

    # load
    d_trainer.load()
    g_trainer.load()

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
            d_trainer.train(100, agent)
            d_trainer.save()
            current_phase = 2

        elif current_phase == 2:
            g_trainer.run(epoch)
            current_phase = 0

        np.savez('saves/state', current_phase=current_phase, epoch=epoch)


if __name__ == "__main__":
    main()
