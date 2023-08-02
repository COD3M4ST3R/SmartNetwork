
# LOCAL IMPORTS
from REPOSITORY.Repository import repository

# IMPORTS
import keras.layers as kl

# CONSTANTS
KERNEL = repository.HYPERPARAMETER["convolution_layer"]["kernel"]
IMG_HEIGHT = repository.INPUT["image"]["dimension"]["height"]
IMG_WIDTH = repository.INPUT["image"]["dimension"]["width"]


def apply(): 
    NotImplemented
    kl.RandomFlip("horizontal_and_vertical", input_shape = (IMG_HEIGHT, IMG_WIDTH, KERNEL))
    kl.RandomRotation(0.1)
    kl.RandomZoom(0.1)
