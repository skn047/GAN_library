import tensorflow as tf


# network module
def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0, 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding="same", kernel_initializer=initializer,
                                      use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0, 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding="same", kernel_initializer=initializer
                                               , use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.down_stack = [
            downsample(64, 4, apply_batchnorm=False),  # (bs, 128, 128, 64)
            downsample(128, 4),  # (bs, 64, 64, 128)
            downsample(256, 4),  # (bs, 32, 32, 256)
            downsample(512, 4),  # (bs, 16, 16, 512)
            downsample(512, 4),  # (bs, 8, 8, 512)
            downsample(512, 4),  # (bs, 4, 4, 512)
            downsample(512, 4),  # (bs, 2, 2, 512)
            downsample(512, 4),  # (bs, 1, 1, 512)
        ]

        self.up_stack = [
            upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
            upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
            upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
            upsample(512, 4),  # (bs, 16, 16, 1024
            upsample(256, 4),  # (bs, 32, 32, 512)
            upsample(128, 4),  # (bs, 64, 64, 256)
            upsample(64, 4),  # (bs, 128, 128, 128)
        ]

        self.last = tf.keras.layers.Conv2DTranspose(3, 4, strides=2, padding='same',
                                                    kernel_initializer=tf.random_normal_initializer(0, 0.02),
                                                    activation='tanh')  # (bs, 256, 256, 3)

    def call(self, inputs, training=True):
        x = inputs

        skips = []
        for down in self.down_stack:
            x = down(x)
            skips.append(x)
        skips = reversed(skips[:-1])

        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])

        output = self.last(x)

        return output


# Discriminator
class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.down_stack = [downsample(64, 4, False), downsample(128, 4), downsample(256, 4)]
        self.zero_pad1 = tf.keras.layers.ZeroPadding2D()
        self.zero_pad2 = tf.keras.layers.ZeroPadding2D()
        self.conv = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=tf.random_normal_initializer(0, 0.02),
                                           use_bias=False)
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.leaky_relu = tf.keras.layers.LeakyReLU()
        self.last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=tf.random_normal_initializer(0, 0.02))

    def call(self, inputs):
        # inputs = input + target(concatenated)
        inp, tar = inputs
        x = tf.concat([inp, tar], 0)
        for down in self.down_stack:
            x = down(x)
        x = self.conv(self.zero_pad1(x))
        x = self.zero_pad2(self.leaky_relu(self.batchnorm(x)))
        output = self.last(x)

        return output
