import cv2
import numpy as np
import tensorflow as tf
from random import uniform, randint, shuffle
from PIL import Image

def imshow(title, img, checker):
    if img.shape[-1] == 1 or img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow(title, img)
        return img

    im = img
    alpha = im[..., -1]
    im = np.delete(im, -1, -1)

    h, w, _ = im.shape

    if h > w:
        longer = h
    else:
        longer = w

    # Make a checkerboard background image same size, dark squares are grey(102), light squares are grey(152)
    if checker:
        bg = np.fromfunction(np.vectorize(lambda i, j: .5 + .25 * ((i+j) % 2)), (int(longer / 25), int(longer / 25)))
    else:
        bg = np.ones([int(longer / 25), int(longer / 25)], dtype=np.float)
    bg = cv2.resize(bg, (longer, longer), interpolation=cv2.INTER_NEAREST)

    # Trim to correct size
    bg = bg[:h, :w]
    # Blend, using result = alpha*overlay + (1-alpha)*background
    im = ((alpha[..., None] * im + (1.0-alpha[..., None]) * bg[..., None]) * 255).astype(np.uint8)
    img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    cv2.imshow(title, img)
    return im


def display_images(images, checker=True):
    pow2val = 1
    pow2 = 0

    while images.shape[0] > pow2val:
        pow2 += 1
        pow2val = pow2 * pow2

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
    img = cv2.resize(img, (img.shape[1] * 5, img.shape[0] * 5), interpolation=cv2.INTER_NEAREST)
    if len(img.shape) == 2:
        img = np.reshape(img, (*img.shape, 1))
    im = imshow('image', img, checker)
    cv2.waitKey(1)
    return im


def generate_noisy_input(src, num=8):
    num_real = num - 1
    random_samples = src[np.random.choice(src.shape[0], num_real)]

    x2 = random_samples[:, :, :, :-1]
    x2 = tf.image.random_hue(x2, 0.5)
    x3 = random_samples[:, :, :, -1:]
    random_samples = tf.concat((x2, x3), -1)
    state = np.random.uniform(0, 1, [1 + num_real, *src[0].shape]).astype("float32")

    for i in range(num_real):
        flip = uniform(0, 1) < 0.5
        s = state[i + 1]
        if flip:
            s = cv2.flip(s, 1)
        r = i / num_real
        state[i + 1] = s * r + random_samples[i] * (1 - r)
    return state


def generate_blotched_input(src, num=8, scale=True):
    result = src[np.random.choice(len(src), num)]

    x2 = result[:, :, :, :-1]
    x2 = tf.image.random_hue(x2, 0.5)
    x3 = result[:, :, :, -1:]
    result = tf.concat((x2, x3), -1).numpy()
    for i in range(num):
        sprite_arr = result[i]

        if num != 1 and scale:
            sprite_arr = blotch(sprite_arr, int(5 * (num - i - 1) / (num - 1)))
        else:
            sprite_arr = blotch(sprite_arr, 5)

        result[i] = sprite_arr
    return result


def blotch(sprite, iterations):
    result = sprite
    indicies = list(np.ndindex((result.shape[0] - 2, result.shape[1] - 2)))
    for _ in range(iterations):
        shuffle(indicies)
        for x, y in indicies:
            # x = randint(1, result.shape[0] - 2)
            # y = randint(1, result.shape[1] - 2)
            x += 1
            y += 1
            n = result[x-1:x+2, y-1:y+2, :]
            n = np.reshape(n, [9, n.shape[-1]])
            nb = np.expand_dims(n, 1)
            w = n == nb
            w = w.all(axis=-1)
            if w.all():
                continue
            w = np.where(w, 1, 0)

            w = np.sum(w, 0)
            w = np.reshape(w, [-1])
            w[4] = 0
            if n.shape[-1] == 4:
                w = np.where((n == [0, 0, 0, 1]).all(axis=-1), 0.1, w)
            else:
                w = np.where((n == [0, 0, 1]).all(axis=-1), 0.1, w)
            w = np.square(w)

            choice = np.random.choice(9, 1, p=[x / sum(w) for x in w])
            result[x, y] = n[choice.item()]
    if np.amax(result) == 0:
        return blotch(sprite, iterations)
    return result


def generate_blotched_demo(src, count=3, steps=5, iter_per_step=1):
    results = []
    for i in range(count):
        im = src[np.random.choice(len(src), 1)][0]
        im_prog = [im.copy()]
        prog = im
        for i in range(steps):
            prog = blotch(prog, iter_per_step)
            im_prog.append(prog.copy())
        results.append(cv2.hconcat(im_prog))
    results = cv2.vconcat(results)

    im = Image.fromarray((results * 255).astype(np.uint8))

    im.save(f"saves/blotch demo.png")
    return results
