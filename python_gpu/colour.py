from PIL import Image
import numpy as np


def hex_to_rgb(hex):
    """
    Convert a hex colour to an RGB colour.
    """

    hex = hex.lstrip("#")
    return tuple(int(hex[i : i + 2], 16) for i in (0, 2, 4))


def build_colour_map(hex_list: list, n: int = 256):
    """
    Build a list of linearly interpolates between a list of RGB tuples.
    """

    rgb_list = [hex_to_rgb(str(hex)) for hex in hex_list]

    if len(rgb_list) == 1:
        return rgb_list * n

    splits = [int(n / (len(rgb_list) - 1))] * (len(rgb_list) - 1)
    for i in range(n - sum(splits)):
        splits[i] += 1

    cmap = []
    for i, split in enumerate(splits):
        r1, g1, b1 = rgb_list[i]
        r2, g2, b2 = rgb_list[i + 1]

        r_step = (r2 - r1) / max(1, split - 1)
        g_step = (g2 - g1) / max(1, split - 1)
        b_step = (b2 - b1) / max(1, split - 1)

        for n in range(split):
            r = int(r1 + (n * r_step))
            g = int(g1 + (n * g_step))
            b = int(b1 + (n * b_step))

            cmap.append((r, g, b))

    return cmap


def image(data, max_iters, cmap):
    """
    Convert data to an image using a colouring function.
    """

    height, width = data.shape
    img = np.zeros((height, width, 3), dtype=np.uint8)

    n = len(cmap)
    for i in range(height):
        for j in range(width):
            img[i, j] = cmap[int(data[i, j] / max_iters * n) % n]

    return img


def encode(data):
    """
    Encode an image as a PNG.
    """

    return Image.fromarray(data)
