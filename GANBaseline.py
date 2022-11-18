from Models import Discriminator, DiscriminatorSN, Discriminator_V2, TD3Agent, DiscriminatorBuffer
import numpy as np
import settings
import tensorflow as tf
import tensorflow_addons as tfa
from utils import display_images, generate_noisy_input, generate_blotched_input
from PIL import Image
import os
from tqdm import tqdm

def pad(x):
    return tf.pad(x, [[0, 0], [0, 1], [11, 12], [0, 0]])


def discriminator_loss(real_output, fake_output, buf_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_buf_loss = cross_entropy(tf.zeros_like(buf_output), buf_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss + fake_buf_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def augment(x):
    x = tf.image.random_flip_left_right(x)
    x2 = x[:, :, :, :-1]
    x2 = tf.image.random_hue(x2, 0.5)
    x3 = x[:, :, :, -1:]
    return tf.concat((x2, x3), -1)

batch_size = 64

# load dataset
data = np.load(settings.dataset_path)

data = data.astype("float32")
data /= 255

data = pad(data).numpy()

dataset = tf.data.Dataset \
    .from_tensor_slices(data) \
    .shuffle(data.shape[0]) \
    .batch(batch_size)\
    .map(lambda x: augment(x),
            num_parallel_calls=tf.data.AUTOTUNE)
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# data = x_train
# data = data.astype("float32")
# data /= 255
#
# dataset = tf.data.Dataset\
#     .from_tensor_slices(data)\
#     .shuffle(data.shape[0])\
#     .batch(batch_size)


for f in dataset.take(1):
    input_shape = f.shape[1:]

# load models

disc = Discriminator(input_shape)
cross_entropy = tf.keras.losses.BinaryCrossentropy()
generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4,  beta_1=0)


gen_input = tf.keras.Input(input_shape)
gen_x = TD3Agent(input_shape).call(gen_input)
gen_out = tf.keras.layers.Conv2D(input_shape[-1], 3, activation="sigmoid", padding="same", use_bias=False)(gen_x)
gen = tf.keras.Model(gen_input, gen_out)
d_buf = DiscriminatorBuffer(10000, input_shape)


@tf.function
def train_step(images, buf, g_in):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = gen(g_in)

        real_output = tf.math.sigmoid(disc(images))
        fake_output = tf.math.sigmoid(disc(generated_images))
        fake_buf_output = tf.math.sigmoid(disc(buf))

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output, fake_buf_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, gen.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, disc.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, gen.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, disc.trainable_variables))
    return generated_images


save_dir = 'saves/GAN Baseline 2'
epoch = 0

d_buf.add(np.random.normal(0, 1, input_shape))
b_i = generate_blotched_input(data, batch_size, False)
while True:
    b_i = np.append(b_i, generate_blotched_input(data, 1), axis=0)
    epoch += 1
    for image_batch in tqdm(dataset):
        f_b = d_buf.sample(batch_size // 2)
        g_in = b_i[np.random.choice(len(b_i), batch_size//2)]
        fake = train_step(image_batch, f_b, g_in)
    for img in fake:
        d_buf.add(img)
    im = display_images(f_b)

    if epoch % 10 != 0:
        continue
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    im = Image.fromarray(im)
    im.save(f"{save_dir}/{epoch}.png")
