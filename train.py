from __future__ import print_function
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import random
import os
import matplotlib.pyplot as plt
import time
import datetime
from model.pix2pix import Generator, Discriminator

# dataset
BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256


def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_png(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [286, 286], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return image


def load_image(image_file):
    image = load(image_file)

    return image


train_target_dataset_path = os.listdir('data/train/')
train_target_dataset_path = [os.path.join(os.path.abspath('data/train/'), p)
                            for p in train_target_dataset_path]

test_target_dataset_path = os.listdir('data/test/')
test_target_dataset_path = [os.path.join(os.path.abspath('data/test/'), p)
                           for p in test_target_dataset_path]

train_aug_dataset_path = os.listdir('data/aug/train')
train_aug_dataset_path = [os.path.join(os.path.abspath('data/aug/train'), p)
                          for p in train_aug_dataset_path]

test_aug_dataset_path = os.listdir('data/aug/test')
test_aug_dataset_path = [os.path.join(os.path.abspath('data/aug/test'), p)
                         for p in test_aug_dataset_path]

train_target_dataset = tf.data.Dataset.from_tensor_slices(train_target_dataset_path)
train_target_dataset = train_target_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

test_target_dataset = tf.data.Dataset.from_tensor_slices(test_target_dataset_path)
test_target_dataset = test_target_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

train_aug_dataset = tf.data.Dataset.from_tensor_slices(train_aug_dataset_path)
train_aug_dataset = train_aug_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

test_aug_dataset = tf.data.Dataset.from_tensor_slices(test_aug_dataset_path)
test_aug_dataset = test_aug_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

train_dataset = tf.data.Dataset.zip((train_aug_dataset, train_target_dataset))
test_dataset = tf.data.Dataset.zip((test_aug_dataset, test_target_dataset))

LAMDA = 100
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target-gen_output))
    total_gen_loss = gan_loss + (LAMDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


log_dir = "logs/"

summary_writer = tf.summary.create_file_writer(
    log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# training
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

generator = Generator()
discriminator = Discriminator()

checkpoint_dir = 'training_checkpoints/'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


@tf.function
def train_step(input_image, target, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
        tf.summary.scalar('disc_loss', disc_loss, step=epoch)


def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image


def random_crop(input_image, real_image):
    stack_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(stack_image, size=[2, -1, 256, 256, 3])

    return cropped_image[0], cropped_image[1]


@tf.function
def random_jitter(input_image, real_image):
    input_image, real_image = resize(input_image, real_image, 256, 256)
    #input_image, real_image = random_crop(input_image, real_image)

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    input_image, real_image = normalize(input_image, real_image)
    return input_image, real_image


@tf.function
def aug_image(input_image, real_image):
    input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


def fit(train_ds, epochs, test_ds):
    for epoch in range(epochs):
        start = time.time()

        display.clear_output(wait=True)

        # for example_input, example_target in test_ds.take(1)

        for n, (input_image, target) in enumerate(train_ds.batch(8).shuffle(8)):
            print('.', end='')
            if (n+1) % 100 == 0:
                print()
            input_image, target = aug_image(input_image, target)
            train_step(input_image, target, epoch)
        print()

        # saving (checkpoint) the model every 20 epochs
        if (epoch) % 20 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))
    checkpoint.save(file_prefix=checkpoint_prefix)


EPOCHS = 100
fit(train_dataset, EPOCHS, test_dataset)
