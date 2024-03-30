from PIL import ImageEnhance, Image
from scipy import ndimage
import numpy as np
import random


def add_some_noise_for_digits(image):
    image.shape = (28, 28)
    # dtype int -> float
    image = image / 1

    # resize image randomly and crop 28x28 image
    temp_file = Image.fromarray(image).convert('L')
    temp_size = np.random.randint(10, 35)
    temp_file = temp_file.resize((temp_size, temp_size))
    image = np.array(temp_file.crop((temp_file.size[0] / 2 - 14, temp_file.size[1] / 2 - 14, temp_file.size[1] / 2 + 14,
                                     temp_file.size[1] / 2 + 14)))

    # rotate image
    temp_file = ndimage.rotate(image, random.randint(-20, 20), reshape=False, prefilter=False) / 1

    # grayscale
    temp_file = Image.fromarray(temp_file).convert("L")

    # change sharpness and bluring
    enhancer = ImageEnhance.Sharpness(temp_file)
    factor = (np.random.random(1)[0] + 0.1) * 5
    temp_file = enhancer.enhance(factor)
    image = np.array(temp_file)

    # shift image along x and y axes
    image = np.roll(image, random.randint(-4, 4), axis=1)
    image = np.roll(image, random.randint(-2, 2), axis=0)

    # add noise
    pixels = [250, 150, 100, 50, 0]
    noise = np.random.choice(pixels, 784, p=[0.003, 0.005, 0.007, 0.015, 0.97])
    noise.shape = (28, 28)
    image = image + (1 * image < 50) * noise
    image.shape = (1, 784)

    return image[0]