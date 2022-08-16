import tensorflow as tf
import tensorflow_addons as tfa


def downsample(filters):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, 4, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))
    result.add(tfa.layers.InstanceNormalization(axis=-1))
    result.add(tf.keras.layers.Activation('relu'))

    result.add(tf.keras.layers.Conv2D(filters, 1, kernel_initializer=initializer, use_bias=False))
    result.add(tfa.layers.InstanceNormalization(axis=-1))
    result.add(tf.keras.layers.Activation('relu'))
    return result


def upsample(filters):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, 4, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))
    result.add(tfa.layers.InstanceNormalization(axis=-1))
    result.add(tf.keras.layers.Activation('relu'))

    result.add(tf.keras.layers.Conv2D(filters, 1, kernel_initializer=initializer, use_bias=False))
    result.add(tfa.layers.InstanceNormalization(axis=-1))
    result.add(tf.keras.layers.Activation('relu'))
    return result


def UNet(x):
    down_stack = [
        downsample(32),
        downsample(64),
        downsample(128),
        downsample(256),
    ]

    up_stack = [
        upsample(256),
        upsample(128),
        upsample(64),
        upsample(32),
        # last upsample must be outside this list
    ]

    last = tf.keras.Sequential()
    last.add(upsample(32))  # last upsample here
    last.add(tf.keras.layers.Conv2D(3, 1, padding='same', activation='tanh'))

    # Downsampling through the models
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)
    return x
