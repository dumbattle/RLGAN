import cv2
import numpy as np
import tensorflow as tf


def imshow(title, img):
    im = img
    alpha = im[..., -1]
    im = np.delete(im, -1, -1)

    h, w, _ = im.shape

    if h > w:
        longer = h
    else:
        longer = w

    # Make a checkerboard background image same size, dark squares are grey(102), light squares are grey(152)
    bg = np.fromfunction(np.vectorize(lambda i, j: .5 + .25 * ((i+j) % 2)), (int(longer / 25), int(longer / 25)))
    bg = cv2.resize(bg, (longer, longer), interpolation=cv2.INTER_NEAREST)

    # Trim to correct size
    bg = bg[:h, :w]
    # Blend, using result = alpha*overlay + (1-alpha)*background
    im = ((alpha[..., None] * im + (1.0-alpha[..., None]) * bg[..., None]) * 255).astype(np.uint8)

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    cv2.imshow(title, im)


def generate_noisy_input(src):
    num_real = 16 - 1
    random_samples = src[np.random.choice(src.shape[0], num_real)]
    state = np.random.uniform(0, 1, [1 + num_real, *src[0].shape]).astype("float32")

    for i in range(num_real):
        state[i + 1] = np.where(
            np.random.uniform(0, 1, src[0].shape) < float(i) / num_real,
            state[i + 1],
            random_samples[i]
        )
    return state


def display_images(images):
    pow2val = 1
    pow2 = 0

    while images.shape[0] > pow2val:
        pow2val *= 2
        pow2 += 1

    img = None
    row = None
    row_count = 0

    for i, im in enumerate(images):
        im = tf.squeeze(tf.clip_by_value(im, 0, 1)).numpy()

        if row is None:
            row = im
            row_count = 1
        else:
            row = cv2.hconcat([row, im])
            row_count += 1

        if row_count == pow2:
            if img is None:
                img = row
            else:
                img = cv2.vconcat([img, row])

            row_count = 0
            row = None
    if row_count != 0:
        im = np.zeros_like(tf.squeeze(images[0]))

        while row_count < pow2:
            row = cv2.hconcat([row, im])
            row_count += 1
        if img is None:
            img = row
        else:
            img = cv2.vconcat([img, row])

    img = cv2.resize(img, (img.shape[0] * 5, img.shape[1] * 5), interpolation=cv2.INTER_NEAREST)
    imshow('image', img)
    cv2.waitKey(1)