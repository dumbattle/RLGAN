import tensorflow as tf
from tensorflow.keras import Sequential, layers
from DenseNet import DenseNet
import numpy as np
import settings
from tqdm import tqdm


def Discriminator(input_shape):
    return Sequential([
        DenseNet(input_shape),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid'),
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


class DiscriminatorTrainer:
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    def __init__(self, train_buffer, real_dataset, discriminator):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
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

            if total_loss < .01:
                return
