import tensorflow as tf
from tensorflow.keras import Sequential, layers
from DenseNet import DenseNet
import numpy as np
import settings
from tqdm import tqdm
import cv2
from utils import imshow


def Discriminator(input_shape):
    return Sequential([
        DenseNet(input_shape),
        layers.Dense(64, activation='relu'),
        layers.Dense(1),
    ])


class DiscriminatorBuffer:
    def __init__(self, size, shape):
        self.images = np.zeros((size, *shape), dtype=np.float32)
        self.count = 0
        self.size = size

    def add(self, image):
        if self.count >= self.size:
            idx = np.random.choice(self.size, 1)
        else:
            idx = self.count
        self.images[idx] = image
        self.count += 1

    def sample(self, batch_size):
        # Get sampling range
        record_range = min(self.count, self.size)

        # Randomly sample indices
        batch_indices = np.random.choice(record_range, batch_size)

        return tf.convert_to_tensor(self.images[batch_indices])

    def clear(self):
        self.count = 0


class DiscriminatorTrainer:
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def __init__(self, train_buffer, real_dataset, discriminator):
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0003)
        self.gen_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0003)
        self.buffer = train_buffer
        self.dataset = real_dataset
        self.disc = discriminator

    def _discriminator_loss(self, real_output, fake_output):
        real_loss = tf.reduce_sum(self.cross_entropy(tf.ones_like(real_output), real_output))
        fake_loss = tf.reduce_sum(self.cross_entropy(tf.zeros_like(fake_output), fake_output))
        total_loss = real_loss + fake_loss
        return total_loss

    def _disc_train_step(self, real, fake):
        with tf.GradientTape() as tape:
            real_out = self.disc(real)
            fake_out = self.disc(fake)

            loss = self._discriminator_loss(real_out, fake_out)
        grads = tape.gradient(loss, self.disc.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.disc.trainable_variables))
        return loss

    def train(self, num_epochs):
        for e in range(num_epochs):
            total_loss = 0
            num = 0

            pbar = tqdm(self.dataset, f"Epoch {e}")

            for batch in pbar:
                num += 1
                fake = self.buffer.sample(settings.Discriminator.Training.batch_size)
                loss = self._disc_train_step(batch, fake)
                total_loss += tf.reduce_mean(loss)
                pbar.set_postfix_str(f"loss: {total_loss / num}")
            pbar.close()

            if total_loss / num < .01:
                self.buffer.count = 0
                return

    def train_disc(self, gen, num_epochs):
        def train_step(real):
            fake = tf.random.uniform((settings.Discriminator.Training.batch_size, *self.disc.input_shape[1:]), 0, 1)
            for _ in range(200):
                action = gen.actor(fake)
                # action = gen.sample(action, 1)
                fake = gen.update_img(fake, action)

            with tf.GradientTape() as tape:
                real_out = self.disc(real)
                fake_out = self.disc(fake)

                step_loss = self._discriminator_loss(real_out, fake_out)

            grads = tape.gradient(step_loss, self.disc.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.disc.trainable_variables))

            return step_loss
        bs = settings.Discriminator.Training.batch_size

        for e in range(num_epochs):
            total_loss = 0
            num = 0

            pbar = tqdm(self.dataset, f"Disc Epoch {e}")

            for batch in pbar:
                num += 1

                loss = train_step(batch)

                total_loss += tf.reduce_mean(loss)
                pbar.set_postfix_str(f"loss: {total_loss / num}")
            pbar.close()

            if total_loss / num < .01:
                self.buffer.count = 0
                return

    def train_gen(self, gen, num_epochs):
        @tf.function
        def train_step(src):
            # generator
            with tf.GradientTape() as tape:
                # create fake
                action = gen.actor(src)
                next_img = gen.update_img(src, action)

                # predict
                fake_out = self.disc(next_img)

                # loss
                gen_loss = tf.reduce_sum(self.cross_entropy(tf.ones_like(fake_out), fake_out))

            # gradient
            grads = tape.gradient(gen_loss, gen.actor.trainable_variables)
            self.gen_optimizer.apply_gradients(zip(grads, gen.actor.trainable_variables))
            return next_img, gen_loss
        bs = settings.Discriminator.Training.batch_size

        for e in range(num_epochs):
            total_loss = 0
            num = 0

            pbar = tqdm(range(200), f"Gen  Epoch {e}")

            # initial state
            img = tf.random.uniform((bs, *self.disc.input_shape[1:]), 0, 1)

            for _ in pbar:
                num += 1
                # train step
                img, loss = train_step(img)

                # update loss
                total_loss = tf.reduce_mean(loss)

                # update pbar
                pbar.set_postfix_str(f"loss: {total_loss / num}")

                # display image
                img2 = tf.squeeze(tf.clip_by_value(img[0], 0, 1)).numpy()
                img2 = cv2.resize(img2, (img2.shape[0] * 5, img2.shape[1] * 5), interpolation=cv2.INTER_NEAREST)
                imshow('image', img2)
                cv2.waitKey(1)

            pbar.close()

            if total_loss / num < .01:
                self.buffer.count = 0
                return
