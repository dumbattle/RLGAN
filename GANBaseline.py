from Models import Discriminator, TD3Agent
import numpy as np
import settings
import tensorflow as tf
from utils import display_images, generate_noisy_input
from PIL import Image
import os


def pad(x):
    return tf.pad(x, [[0, 0], [0, 1], [11, 12], [0, 0]])


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# load dataset
data = np.load(settings.dataset_path)

data = data.astype("float32")
data /= 255

data = pad(data).numpy()

dataset = tf.data.Dataset \
    .from_tensor_slices(data) \
    .shuffle(data.shape[0]) \
    .batch(settings.Discriminator.Training.batch_size)

for f in dataset.take(1):
    input_shape = f.shape[1:]

# load models

disc = Discriminator(input_shape)
gen_input = tf.keras.Input(input_shape)
gen_x = TD3Agent(input_shape).call(gen_input)
gen_out = tf.keras.layers.Conv2D(input_shape[-1], 1, activation="sigmoid", padding="same", use_bias=False)(gen_x) * 1.1
gen = tf.keras.Model(gen_input, gen_out)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


@tf.function
def train_step(images, noise):

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = gen(noise)

        real_output = disc(images)
        fake_output = disc(generated_images)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, gen.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, disc.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, gen.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, disc.trainable_variables))
    return generated_images


epoch = 0
while True:
    epoch += 1
    for image_batch in dataset:
        noise = generate_noisy_input(data, settings.Discriminator.Training.batch_size)
        fake = train_step(image_batch, noise)
        fake = display_images(fake)
    if epoch % 10 != 0:
        continue
    if not os.path.exists(f'saves/GAN Baseline'):
        os.makedirs(f'saves/GAN Baseline')

    im = Image.fromarray(fake)
    im.save(f"saves/GAN Baseline/{epoch}.png")
