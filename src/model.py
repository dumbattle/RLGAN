import random

import tensorflow as tf
import tensorflow.keras.layers as Layers
import numpy as np


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = Layers.Conv2D(32, (5, 5), strides=(1, 1), activation="relu", input_shape=[None, 31, 41, 4])
        self.conv2 = Layers.Conv2D(64, (5, 5), strides=(1, 1), activation="relu")
        self.pool = Layers.GlobalMaxPool2D()
        self.fc1 = Layers.Dense(16, activation="relu")
        self.fc2 = Layers.Dense(1)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x


class RLGenerator(tf.keras.Model):
    def __init__(self):
        super(RLGenerator, self).__init__()
        self.conv1 = Layers.Conv2D(32, (5, 5), activation="relu", padding="same", input_shape=[None, 31, 41, 4])
        self.conv2 = Layers.Conv2D(64, (5, 5), activation="relu", padding="same")
        self.conv3 = Layers.Conv2D(128, (5, 5), activation="relu", padding="same")

        # actor
        self.mean = Layers.Conv2D(4, (5, 5), activation="tanh", padding="same")
        self.std_dev = Layers.Conv2D(4, (5, 5), activation="tanh", padding="same")

        # critic
        self.pool = Layers.GlobalMaxPool2D()
        self.fc1 = Layers.Dense(16, activation="relu")
        self.fc2 = Layers.Dense(1)

    def call(self, x):
        src = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        mean = self.mean(x) * 5.0 / 255
        std_dev = self.std_dev(x) * 5.0 / 255

        action = tf.random.normal(mean.shape, mean, std_dev)
        probs = tf.exp(-tf.pow(action - mean, 2) / (2 * std_dev * std_dev + 1e-9)) / \
            (tf.sqrt(2 * 3.1415) * std_dev + 1e-9)
        result = tf.identity(src + action)

        critic = self.pool(x)
        critic = self.fc1(critic)
        critic = self.fc2(critic)
        return action, probs, result, critic
