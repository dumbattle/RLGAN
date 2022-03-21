import tensorflow as tf


class ConvSelfAttn(tf.keras.Model):
    def __init__(self, num_input_channels):
        super(ConvSelfAttn, self).__init__()
        self.num_input_channels = num_input_channels
        self.attn_weight = self.add_weight(self.name + '_attn_weight',
                                           shape=(),
                                           initializer=tf.initializers.Zeros)
        self.q_conv = tf.keras.layers.Conv2D(filters=num_input_channels // 8,
                                             kernel_size=1)
        self.k_conv = tf.keras.layers.Conv2D(filters=num_input_channels // 8,
                                             kernel_size=1)
        self.v_conv = tf.keras.layers.Conv2D(filters=num_input_channels,
                                             kernel_size=1)

    def call(self, inputs):
        q = self.q_conv(inputs)
        k = self.k_conv(inputs)
        v = self.v_conv(inputs)
        attn = self._attn(q, k, v)
        return attn * self.attn_weight + inputs

    @staticmethod
    def _attn(q, k, v):
        input_shape = tf.shape(q)

        batchsize = input_shape[0]
        height = input_shape[-3]
        width = input_shape[-2]
        q2 = tf.reshape(q, (batchsize, height * width, -1))
        k2 = tf.transpose(
            tf.reshape(k, (batchsize, height * width, -1)), (0, 2, 1))
        v2 = tf.transpose(
            tf.reshape(v, (batchsize, height * width, -1)), (0, 2, 1))

        attn = tf.matmul(q2, k2)
        attn = tf.nn.softmax(attn)
        result = tf.matmul(v2, attn, transpose_b=True)

        result = tf.transpose(result, (0, 2, 1))
        result = tf.reshape(result, (batchsize, height, width, -1))

        return result

    def get_config(self):
        return {'num_input_channels': self.num_input_channels}

    @classmethod
    def from_config(cls, config):
        return ConvSelfAttn(config['num_input_channels'])


tf.keras.utils.get_custom_objects().update({'ConvSelfAttn': ConvSelfAttn})

if __name__ == '__main__':
    test_input = tf.zeros((16, 41, 31, 4))
    attn_model = ConvSelfAttn(4)
