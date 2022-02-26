from tensorflow.keras import Model, layers

import tensorflow as tf
import settings


def dense_block(inp, num_channels):
    for i in range(settings.DenseNet.layers_per_block):
        # bottleneck
        x = layers.BatchNormalization()(inp)
        x = layers.Activation(settings.DenseNet.activation)(x)
        x = layers.Conv2D(4 * settings.DenseNet.growth_rate, 1, use_bias=False,)(x)

        # conv
        x = layers.BatchNormalization()(x)
        x = layers.Activation(settings.DenseNet.activation)(x)
        x = layers.Conv2D(settings.DenseNet.growth_rate, 3, padding='same', use_bias=False,)(x)

        # concat
        inp = layers.Concatenate()([inp, x])
        num_channels += settings.DenseNet.growth_rate

    return x, num_channels


def transition(x, num_channels):
    x = layers.BatchNormalization()(x)
    x = layers.Activation(settings.DenseNet.activation)(x)

    x = layers.Conv2D(settings.DenseNet.growth_rate, 3, padding='same', use_bias=False, )(x)
    num_channels = int(num_channels * settings.DenseNet.reduction)
    x = layers.Conv2D(num_channels, 1, use_bias=False)(x)
    x = layers.AveragePooling2D(2, strides=2)(x)
    return x, num_channels


def DenseNet(input_shape, pool=True):
    inp = layers.Input(shape=input_shape)
    x = inp
    num_channels = input_shape[-1]
    for _ in range(settings.DenseNet.num_blocks):
        x, num_channels = dense_block(x, num_channels)

    x = layers.BatchNormalization()(x)
    x = layers.Activation(settings.DenseNet.activation)(x)

    if pool:
        x = layers.GlobalAveragePooling2D()(x)

    return Model(inputs=inp, outputs=x)
