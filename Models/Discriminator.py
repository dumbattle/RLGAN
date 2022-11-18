import tensorflow as tf
from tensorflow.keras import Sequential, layers
from DenseNet import DenseNet
import numpy as np
import settings
from tqdm import tqdm
import os
import tensorflow_addons as tfa
from Attention import ConvSelfAttn
from utils import generate_blotched_input, blotch


def Discriminator_V2(input_shape):
    inp = tf.keras.Input(shape=input_shape)
    # block 1
    x = layers.Conv2D(64, 3, padding='same', use_bias=False)(inp)
    x = tfa.layers.InstanceNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)
    x = layers.AveragePooling2D(2, strides=2)(x)
    # block 2
    x = layers.Conv2D(128, 3, padding='same', use_bias=False)(x)
    x = tfa.layers.InstanceNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)
    x = layers.AveragePooling2D(2, strides=2)(x)
    # block 3
    x = layers.Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = tfa.layers.InstanceNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)
    x = layers.AveragePooling2D(2, strides=2)(x)
    # block 4
    x = layers.Conv2D(512, 3, padding='same', use_bias=False)(x)
    x = tfa.layers.InstanceNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)
    x = layers.AveragePooling2D(2, strides=2)(x)
    # x = ConvSelfAttn(512)(x)
    # block 5
    x = layers.Conv2D(512, 3, padding='same', use_bias=False)(x)
    x = tfa.layers.InstanceNormalization(axis=-1)(x)

    x = layers.Activation('relu')(x)
    # Dsicriminator head
    x = layers.GlobalAvgPool2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(1)(x)

    x2 = layers.Conv2D(64, 3, padding='same')(inp)
    x2 = tfa.layers.InstanceNormalization(axis=-1)(x2)
    x2 = layers.Activation('relu')(x2)

    x2 = layers.Conv2D(1, 1, padding='same')(x2)
    x2 = tfa.layers.InstanceNormalization(axis=-1)(x2)
    x2 = layers.Activation('relu')(x2)

    x2 = tf.reduce_min(x2, [-1, -2, -3])
    x2 = tf.expand_dims(x2, -1)

    return tf.keras.Model(inputs=inp, outputs=tf.minimum(x, x2))


def Discriminator(input_shape):
    inp = tf.keras.Input(shape=input_shape)
    # block 1
    x = layers.Conv2D(64, 3, padding='same', use_bias=False)(inp)
    x = tfa.layers.InstanceNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)
    x = layers.AveragePooling2D(2, strides=2)(x)
    # block 2
    x = layers.Conv2D(128, 3, padding='same', use_bias=False)(x)
    x = tfa.layers.InstanceNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)
    x = layers.AveragePooling2D(2, strides=2)(x)
    # x = ConvSelfAttn(128)(x)
    # block 3
    x = layers.Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = tfa.layers.InstanceNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)
    x = layers.AveragePooling2D(2, strides=2)(x)
    # x = ConvSelfAttn(256)(x)
    # block 4
    x = layers.Conv2D(512, 3, padding='same', use_bias=False)(x)
    x = tfa.layers.InstanceNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)
    x = layers.AveragePooling2D(2, strides=2)(x)
    # x = ConvSelfAttn(512)(x)
    # block 5
    x = layers.Conv2D(512, 3, padding='same', use_bias=False)(x)
    x = tfa.layers.InstanceNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)

    # Dsicriminator head
    x = layers.GlobalAvgPool2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(1)(x)

    return tf.keras.Model(inputs=inp, outputs=x)


def DiscriminatorSN(input_shape):
    inp = tf.keras.Input(shape=input_shape)
    x = tfa.layers.SpectralNormalization(layers.Conv2D(64,  3, padding='same', use_bias=False, activation='swish'))(inp)
    x = layers.AveragePooling2D(2, strides=2)(x)
    x = tfa.layers.SpectralNormalization(layers.Conv2D(128, 3, padding='same', use_bias=False, activation='swish'))(x)
    x = layers.AveragePooling2D(2, strides=2)(x)
    x = tfa.layers.SpectralNormalization(layers.Conv2D(256, 3, padding='same', use_bias=False, activation='swish'))(x)
    x = layers.AveragePooling2D(2, strides=2)(x)
    x = tfa.layers.SpectralNormalization(layers.Conv2D(512, 3, padding='same', use_bias=False, activation='swish'))(x)
    x = layers.AveragePooling2D(2, strides=2)(x)
    x = tfa.layers.SpectralNormalization(layers.Conv2D(512, 3, padding='same', use_bias=False, activation='swish'))(x)
    x = layers.GlobalAvgPool2D()(x)
    x = tfa.layers.SpectralNormalization(layers.Dense(1))(x)
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

    def save(self, save_path):
        np.savez(save_path, images=self.images, count=self.count)

    def load(self, save_path):
        data = np.load(save_path)
        self.images = data['images']
        self.count = data['count']

        data.close()


class DiscriminatorTrainer:
    loss = tf.keras.losses.MeanSquaredError()

    def __init__(self, train_buffer, real_dataset, discriminator, save_dir, data):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
        self.gen_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
        self.buffer = train_buffer
        self.dataset = real_dataset
        self.data = data
        self.disc = discriminator

        self.save_dir = save_dir

    def save(self):
        if not os.path.exists(f'{self.save_dir}/discriminator'):
            os.makedirs(f'{self.save_dir}/discriminator')
        self.buffer.save(f"{self.save_dir}/discriminator/buffer")

        opt_weights = self.optimizer.get_weights()
        np.savez(f'{self.save_dir}/discriminator/optimizer', *opt_weights)

        self.disc.save_weights(f'{self.save_dir}/discriminator/model/model')

    def load(self):
        if not os.path.exists(f'{self.save_dir}/discriminator/optimizer.npz'):
            return

        # buffer
        self.buffer.load(f"{self.save_dir}/discriminator/buffer.npz")

        # optimizer
        opt_weights_data = np.load(f'{self.save_dir}/discriminator/optimizer.npz')
        opt_weights = [opt_weights_data[x] for x in opt_weights_data.files]
        self._disc_train_step(
            self.buffer.sample(settings.Discriminator.Training.batch_size//2),
            self.buffer.sample(settings.Discriminator.Training.batch_size//2),
            self.buffer.sample(settings.Discriminator.Training.batch_size//2),
            self.buffer.sample(settings.Discriminator.Training.batch_size//2),
            tf.constant([1]*(settings.Discriminator.Training.batch_size//2)))
        self.optimizer.set_weights(opt_weights)
        opt_weights_data.close()

        # discriminator - must be after optimizer
        self.disc.load_weights(f'{self.save_dir}/discriminator/model/model')

    def _discriminator_loss(self, real_real_output, real_noise_output, real_noise_labels, fake_output):
        real_noise_output = tf.squeeze(real_noise_output)
        real_noise_labels = tf.squeeze(real_noise_labels)
        real_real_loss = tf.reduce_sum(self.loss(tf.ones_like(real_real_output), real_real_output))
        real_noise_loss = tf.reduce_sum(self.loss(real_noise_labels, real_noise_output)) * 0.01
        fake_loss = tf.reduce_sum(self.loss(tf.zeros_like(fake_output), fake_output))
        total_loss = real_noise_loss + fake_loss + real_real_loss
        return total_loss,  fake_loss + real_real_loss

    @tf.function()
    def _disc_train_step(self, real, fake_buf, fake_gen, real_noise, noise_val):
        fake = tf.concat([fake_buf, fake_gen], 0)

        with tf.GradientTape() as tape:
            real_real_out = self.disc(real, training=True)
            real_noise_out = self.disc(real_noise, training=True)
            fake_out = self.disc(fake, training=True)

            loss, tracked_loss = self._discriminator_loss(real_real_out, real_noise_out, 1 - noise_val, fake_out)

        grads = tape.gradient(loss, self.disc.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.disc.trainable_variables))
        return tracked_loss

    def train(self, num_epochs, gen):
        for e in range(num_epochs):
            total_loss = 0
            num = 0

            pbar = tqdm(self.dataset, f"Epoch {e}")
            for batch in pbar:
                num += 1
                half_batch = int(settings.Discriminator.Training.batch_size / 2)

                rr = batch[:batch.shape[0] // 2]
                rn = batch[batch.shape[0] // 2:].numpy()

                fake_buf = self.buffer.sample(half_batch)
                fake_gen = gen.generate(count=half_batch, img=generate_blotched_input(self.data))
                n = tf.random.uniform([half_batch], 0, 1)
                for i in range(half_batch):
                    rn[i] = blotch(rn[i], int(5 * n[i]))
                loss = self._disc_train_step(rr, fake_buf, fake_gen, rn, n)

                total_loss += loss.numpy()

                pbar.set_postfix_str(f"loss: {total_loss / num}")
                if num % 64 == 0:
                    if total_loss / num < .001:
                        break
                    num = 0
                    total_loss = 0

            pbar.close()

            if total_loss / num < .001:
                self.save()
                return


if __name__ == '__main__':
    test_input = tf.random.normal((1, 64, 64, 4))

    disc = Discriminator((64, 64, 4))

    disc(test_input)
    disc.summary()
