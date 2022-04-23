import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.losses as losses
from tensorflow.keras import Model, Sequential
import numpy as np
import settings
import cv2
from DenseNet import dense_block
from utils import imshow
from tqdm import tqdm

def main():
    data = np.load(settings.dataset_path)
    data = data.astype("float32")
    data /= 255

    dataset = tf.data.Dataset\
        .from_tensor_slices(data)\
        .shuffle(data.shape[0])\
        .batch(128)

    test_data = data[0]
    test_data = tf.expand_dims(test_data, 0)

    pixel_ae = PixelAE()

    ae = pixel_ae.get_ae()

    # ae.compile(optimizer="rmsprop", loss="binary_crossentropy")
    # ae.fit(
    #     x=data,
    #     y=data,
    #     epochs=50,
    #     batch_size=128,
    #     shuffle=True,
    # )
    # predictions = pixel_ae.decoder(pixel_ae.encoder(test_data))
    # display(predictions)
    # return
    opt = optimizers.RMSprop()
    loss = losses.BinaryCrossentropy()

    @tf.function
    def train_step(sample):
        with tf.GradientTape() as tape:
            out = ae(sample)
            l2 = loss(sample, out)
        variables = ae.trainable_variables
        grads = tape.gradient(l2, variables)
        opt.apply_gradients(zip(grads, variables))
        return l2

    for i in range(1000):
        with tqdm(dataset, f"Epoch: {i}") as pbar:
            for x in pbar:
                total_loss = 0
                count = 0
                l = train_step(x)
                total_loss += l.numpy()
                count += 1
                pbar.set_postfix_str(f"Loss: {total_loss / count}")
        predictions = ae(tf.random.uniform(test_data.shape, 0, 1))
        predictions = ae(test_data)
        display(predictions)


class PixelAE:
    def __init__(self):
        self.encoder = PixelAE.create_encoder()
        self.decoder = PixelAE.create_decoder()
        self.palette_encoder = None
        self.palette_decoder = None

    def get_ae(self):
        inp = layers.Input(shape=(None, None, 4))
        a, b, c = self.encoder(inp)
        d = self.decoder((a, b, c))
        return Model(inp, d)

    @staticmethod
    def create_encoder():
        inp = layers.Input(shape=(None, None, 4))
        x = dense_block(inp, 4, num_layers=6, growth_rate=6, self_attention=False)[0]
        encoded = layers.Conv2D(64, 3, padding='same', activation='relu')(x)

        p = layers.GlobalAveragePooling2D()(x)
        p = layers.Dense(256, activation='relu')(p)

        pk = layers.Dense(settings.PixelAE.palette_size * settings.PixelAE.palette_code_size, activation='relu')(p)
        pk = layers.Reshape((settings.PixelAE.palette_code_size, settings.PixelAE.palette_size))(pk)

        pv = layers.Dense(settings.PixelAE.palette_size * 4, activation='relu')(p)
        pv = layers.Reshape((settings.PixelAE.palette_size, 4))(pv)
        return Model(inputs=[inp], outputs=[encoded, pk, pv])

    @staticmethod
    def create_decoder():
        inp = layers.Input(shape=(None, None, 64))
        pk_inp = layers.Input(shape=(settings.PixelAE.palette_code_size, settings.PixelAE.palette_size))
        pv_inp = layers.Input(shape=(settings.PixelAE.palette_size, 4))

        x = layers.GaussianNoise(1)(inp)
        x = dense_block(x, 64, num_layers=6, growth_rate=6, self_attention=False)[0]
        x = layers.Conv2D(settings.PixelAE.palette_code_size, 3, padding='same', activation='sigmoid')(x)

        pk = tf.expand_dims(pk_inp, -3)
        pv = tf.expand_dims(pv_inp, -3)
        qk = tf.matmul(x, pk)  # (w x h x num_col)
        qk = tf.math.softmax(qk)

        output = tf.matmul(qk, pv)

        return Model(inputs=[inp, pk_inp, pv_inp], outputs=[output])


def display(img):
    img = tf.clip_by_value(tf.squeeze(img), 0, 1).numpy()
    img = cv2.resize(img, (img.shape[0] * 5, img.shape[1] * 5), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('image', img)
    cv2.waitKey(1)


if __name__ == "__main__":
    main()
