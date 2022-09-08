import tensorflow as tf
from tensorflow.keras import Sequential, layers
from DenseNet import DenseNet
import numpy as np
import settings
from tqdm import tqdm
import os
import tensorflow_addons as tfa
from Attention import ConvSelfAttn


def Discriminator(input_shape):
    inp = tf.keras.Input(shape=input_shape)
    # block 1
    x = layers.Conv2D(64, 3, padding='same', use_bias=False)(inp)
    # x = layers.Activation('relu')(x)
    # x = layers.Conv2D(64, 1, use_bias=False)(x)
    x = tfa.layers.InstanceNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)
    x = layers.AveragePooling2D(2, strides=2)(x)
    # block 2
    x = layers.Conv2D(128, 3, padding='same', use_bias=False)(x)
    # x = layers.Activation('relu')(x)
    # x = layers.Conv2D(128, 1, use_bias=False)(x)
    x = tfa.layers.InstanceNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)
    x = layers.AveragePooling2D(2, strides=2)(x)
    # block 3
    x = layers.Conv2D(256, 3, padding='same', use_bias=False)(x)
    # x = layers.Activation('relu')(x)
    # x = layers.Conv2D(256, 1, use_bias=False)(x)
    x = tfa.layers.InstanceNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)
    x = layers.AveragePooling2D(2, strides=2)(x)
    # block 4
    x = layers.Conv2D(512, 3, padding='same', use_bias=False)(x)
    # x = layers.Activation('relu')(x)
    # x = layers.Conv2D(512, 1, use_bias=False)(x)
    x = tfa.layers.InstanceNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)
    x = layers.AveragePooling2D(2, strides=2)(x)
    x = ConvSelfAttn(512)(x)
    # block 5
    x = layers.Conv2D(512, 3, padding='same', use_bias=False)(x)
    # x = layers.Activation('relu')(x)
    # x = layers.Conv2D(512, 1, use_bias=False)(x)
    x = tfa.layers.InstanceNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)
    # Dsicriminator head
    x = layers.GlobalAvgPool2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(1)(x)

    # x = DenseNet(input_shape, pool=False)(inp)
    # x = ConvSelfAttn(x.shape[-1])(x)
    # x = layers.Dense(1)(x)
    # x = tf.reduce_mean(x, (1, 2, 3))
    return tf.keras.Model(inputs=inp, outputs=x)


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

    def save(self):
        if not os.path.exists('saves/discriminator'):
            os.makedirs('saves/discriminator')
        np.savez("saves/discriminator/buffer", images=self.images, count=self.count)

    def load(self):
        if not os.path.exists("saves/discriminator/buffer.npz"):
            return

        data = np.load("saves/discriminator/buffer.npz")
        self.images = data['images']
        self.count = data['count']

        data.close()


class DiscriminatorTrainer:
    loss = tf.keras.losses.MeanSquaredError()

    def __init__(self, train_buffer, real_dataset, discriminator):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
        self.gen_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
        self.buffer = train_buffer
        self.dataset = real_dataset
        self.disc = discriminator

    def save(self):
        if not os.path.exists('saves/discriminator/model'):
            os.makedirs('saves/discriminator/model')
        self.buffer.save()

        opt_weights = self.optimizer.get_weights()
        np.savez("saves/discriminator/optimizer", *opt_weights)

        self.disc.save_weights('saves/discriminator/model/model')

    def load(self):
        if not os.path.exists("saves/discriminator/optimizer.npz"):
            return

        # buffer
        self.buffer.load()

        # optimizer
        opt_weights_data = np.load("saves/discriminator/optimizer.npz")
        opt_weights = [opt_weights_data[x] for x in opt_weights_data.files]
        self._disc_train_step(self.buffer.sample(2), self.buffer.sample(1), self.buffer.sample(1))
        self.optimizer.set_weights(opt_weights)
        opt_weights_data.close()

        # discriminator - must be after optimizer
        self.disc.load_weights('saves/discriminator/model/model')

    def _discriminator_loss(self, real_output, fake_output):
        real_loss = tf.reduce_sum(self.loss(tf.ones_like(real_output), real_output))
        fake_loss = tf.reduce_sum(self.loss(tf.zeros_like(fake_output), fake_output))
        total_loss = real_loss + fake_loss
        return total_loss

    @tf.function()
    def _disc_train_step(self, real, fake_buf, fake_gen):
        fake = tf.concat([fake_buf, fake_gen], 0)

        with tf.GradientTape() as tape:
            real_out = self.disc(real, training=True)
            fake_out = self.disc(fake, training=True)

            loss = self._discriminator_loss(real_out, fake_out)

        grads = tape.gradient(loss, self.disc.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.disc.trainable_variables))
        return loss

    def train(self, num_epochs, gen):
        for e in range(num_epochs):
            total_loss = 0
            num = 0

            pbar = tqdm(self.dataset, f"Epoch {e}")

            for batch in pbar:
                num += 1
                half_batch = int(settings.Discriminator.Training.batch_size / 2)

                fake_buf = self.buffer.sample(half_batch)
                fake_gen = gen.generate(count=half_batch)
                loss = self._disc_train_step(batch, fake_buf, fake_gen)

                total_loss += loss.numpy()
                pbar.set_postfix_str(f"loss: {total_loss / num}")
            pbar.close()

            if total_loss / num < .001:
                self.save()
                return


if __name__ == '__main__':
    test_input = tf.random.normal((1, 64, 64, 4))

    disc = Discriminator((64, 64, 4))

    disc(test_input)
    disc.summary()
