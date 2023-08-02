
# LOCAL IMPORTS
from REPOSITORY.Repository import repository
import AUGMENTATION.Multiplier as a

# IMPORTS
import tensorflow
import keras.layers as kl

# CONSTANTS
SCALE = repository.HYPERPARAMETER["scale"]
FILTER = repository.HYPERPARAMETER["convolution_layer"]["filter"]
KERNEL = repository.HYPERPARAMETER["convolution_layer"]["kernel"]
NEURONS = repository.HYPERPARAMETER["dense"]["neurons"]
IMG_HEIGHT = repository.INPUT["image"]["dimension"]["height"]
IMG_WIDTH = repository.INPUT["image"]["dimension"]["width"]
DROPOUT = repository.HYPERPARAMETER["dropout"]
ACTIVATION_FUNCTION = repository.ACTIVATION_FUNCTION["name"]


def sequential(num_classes: int) -> tensorflow.keras.Sequential:
    model = tensorflow.keras.Sequential([
        # Data Augmentation
        kl.RandomFlip("horizontal_and_vertical", input_shape = (IMG_HEIGHT, IMG_WIDTH, KERNEL)),
        kl.RandomRotation(0.1),
        kl.RandomZoom(0.1),
                
        kl.Rescaling(SCALE, input_shape = (IMG_HEIGHT, IMG_WIDTH, KERNEL)),
        kl.Conv2D(FILTER, KERNEL, activation = ACTIVATION_FUNCTION),
        kl.MaxPooling2D(),
        kl.Conv2D(FILTER, KERNEL, activation = ACTIVATION_FUNCTION),
        kl.MaxPooling2D(),
        kl.Conv2D(FILTER, KERNEL, activation = ACTIVATION_FUNCTION),
        kl.MaxPooling2D(),
        kl.Dropout(DROPOUT),
        kl.Flatten(),
        kl.Dense(NEURONS, activation = ACTIVATION_FUNCTION),
        kl.Dense(num_classes)])
    
    return model
