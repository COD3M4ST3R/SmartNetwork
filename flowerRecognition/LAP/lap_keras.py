
# LOCAL IMPORTS
from CLASSES.Dataset import Dataset 
from REPOSITORY.Repository import repository

# IMPORTS
import tensorflow 
import pathlib
import os
import PIL
import PIL.Image
import matplotlib.pyplot as plt
import random



# CONSTANTS
SEED = repository.DATASET["raw"]["seed"]
SPLIT = repository.DATASET["raw"]["split"]
URL = repository.DATASET["raw"]["url"]

BATCH_SIZE = repository.HYPERPARAMETER["batch"]["size"]
IMG_HEIGHT = repository.INPUT["image"]["dimension"]["height"]
IMG_WEIGHT = repository.INPUT["image"]["dimension"]["width"]

TRAIN_SUBSET = "training"
VALIDATE_SUBSET = "validation"  

AUTOTUNE = tensorflow.data.AUTOTUNE # Automatically Sets the Hyperparameters


def start():
    url = URL
    archive = tensorflow.keras.utils.get_file(origin = url, extract = True)
    directory = pathlib.Path(archive).with_suffix('')
    images = list(directory.glob('*/*'))

    random.seed(SEED)
    random.shuffle(images)

    # TRAIN DATASET ALLIGNMENT
    train = tensorflow.keras.utils.image_dataset_from_directory(
        directory,
        validation_split = SPLIT,
        subset = TRAIN_SUBSET,
        seed = SEED,
        image_size = (IMG_HEIGHT, IMG_WEIGHT),
        batch_size = BATCH_SIZE
    )

    classes = train.class_names

    # VALIDATE DATESET ALLIGNMENT
    validate = tensorflow.keras.utils.image_dataset_from_directory(
        directory, # The directory path containing the dataset
        validation_split = SPLIT, # Float value between 0 and 1, representing the fraction of the dataset to use for validation
        subset = VALIDATE_SUBSET, # String indication
        seed = SEED, # Reproducible  shuffling of the dataset. With this value, we verify that random data has been choosen and it will choose those data on different models or machines so we can compare the performance
        image_size = (IMG_HEIGHT, IMG_WEIGHT), # Target height and weight of each data
        batch_size = BATCH_SIZE # Indicates how many data will be processed at the same time which increases the performance of model while learning. Higher the value: more accurate estimate but less frequent weight update; lower the value: less accurate but more frequent weight update. Need to observe results to obtain the most efficient value.
    )


    dataset = Dataset(classes = classes, train = train, validate = validate)
    
    return dataset