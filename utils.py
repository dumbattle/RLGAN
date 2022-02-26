import cv2
import numpy as np


def imshow(title, img):
    im = img
    alpha = im[..., -1]
    im = np.delete(im, -1, -1)

    h, w, _ = im.shape

    # Make a checkerboard background image same size, dark squares are grey(102), light squares are grey(152)
    bg = np.fromfunction(np.vectorize(lambda i, j: .5 + .25 * ((i+j) % 2)), (16, 16))

    # Resize to square same length as longer side (so squares stay square), then trim
    if h > w:
        longer = h
    else:
        longer = w
    bg = cv2.resize(bg, (longer, longer), interpolation=cv2.INTER_NEAREST)
    # Trim to correct size
    bg = bg[:h, :w]
    # Blend, using result = alpha*overlay + (1-alpha)*background
    im = ((alpha[..., None] * im + (1.0-alpha[..., None]) * bg[..., None]) * 255).astype(np.uint8)
    cv2.imshow(title, im)
