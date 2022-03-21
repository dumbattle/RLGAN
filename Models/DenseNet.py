from tensorflow.keras import Model, layers

import tensorflow as tf
import settings
from Attention import ConvSelfAttn


def dense_block(inp, num_channels, num_layers=None, growth_rate=None, activation=None, self_attention=False):
    if num_layers is None:
        num_layers = settings.DenseNet.layers_per_block

    if growth_rate is None:
        growth_rate = settings.DenseNet.growth_rate

    if activation is None:
        activation = settings.DenseNet.activation

    x = None
    for i in range(num_layers):
        # bottleneck
        x = layers.BatchNormalization()(inp)
        x = layers.Activation(activation)(x)
        x = layers.Conv2D(4 * growth_rate, 1, use_bias=False)(x)

        # conv
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        x = layers.Conv2D(growth_rate, 3, padding='same', use_bias=False)(x)

        # concat
        inp = layers.Concatenate()([inp, x])
        num_channels += growth_rate
    x = inp
    if self_attention:
        x = ConvSelfAttn(num_channels)(inp)
    return x, num_channels


def transition(x, num_channels):
    num_channels = int(num_channels * settings.DenseNet.reduction)

    x = layers.BatchNormalization()(x)
    x = layers.Activation(settings.DenseNet.activation)(x)

    x = layers.Conv2D(settings.DenseNet.growth_rate, 3, padding='same', use_bias=False)(x)
    x = layers.Conv2D(num_channels, 1, use_bias=False)(x)
    x = layers.AveragePooling2D(2, strides=2)(x)

    return x, num_channels


def DenseNet(input_shape, pool=True):
    inp = layers.Input(shape=input_shape)
    x = inp
    num_channels = input_shape[-1]
    for b in range(settings.DenseNet.num_blocks):
        x, num_channels = dense_block(x, num_channels, self_attention=b == settings.DenseNet.num_blocks - 1)
        x, num_channels = transition(x, num_channels)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(settings.DenseNet.activation)(x)

    if pool:
        x = layers.GlobalAveragePooling2D()(x)

    return Model(inputs=inp, outputs=x)
