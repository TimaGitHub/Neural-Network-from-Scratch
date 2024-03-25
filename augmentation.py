from PIL import ImageEnhance, Image
from scipy import ndimage
import numpy as np
import random


def add_some_noise_for_digits(image):
    image.shape = (28, 28)

    image = image / 1

    temp_file = Image.fromarray(image).convert('L')
    temp_size = np.random.randint(10, 35)
    temp_file = temp_file.resize((temp_size, temp_size))

    image = np.array(temp_file.crop((temp_file.size[0] / 2 - 14, temp_file.size[1] / 2 - 14, temp_file.size[1] / 2 + 14,
                                     temp_file.size[1] / 2 + 14)))

    temp_file = ndimage.rotate(image, random.randint(-20, 20), reshape=False, prefilter=False) / 1
    temp_file = Image.fromarray(temp_file).convert("L")

    enhancer = ImageEnhance.Sharpness(temp_file)
    factor = (np.random.random(1)[0] + 0.1) * 5
    temp_file = enhancer.enhance(factor)
    image = np.array(temp_file)

    image = np.roll(image, random.randint(-4, 4), axis=1)
    image = np.roll(image, random.randint(-2, 2), axis=0)

    pixels = [250, 150, 100, 50, 0]
    noise = np.random.choice(pixels, 784, p=[0.003, 0.005, 0.007, 0.015, 0.97])

    noise.shape = (28, 28)

    image = image + (1 * image < 50) * noise

    image.shape = (1, 784)

    return image[0]