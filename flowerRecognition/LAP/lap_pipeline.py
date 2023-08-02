
# LOCAL IMPORTS
from REPOSITORY.Repository import repository
from CLASSES.Dataset import Dataset

# IMPORTS
import tensorflow
import pathlib
import os
import numpy 

# CONSTANTS
SEED = repository.DATASET["raw"]["seed"]
SPLIT = repository.DATASET["raw"]["split"]
URL = repository.DATASET["raw"]["url"]

BATCH_SIZE = repository.HYPERPARAMETER["batch"]["size"]
IMG_HEIGHT = repository.INPUT["image"]["dimension"]["height"]
IMG_WEIGHT = repository.INPUT["image"]["dimension"]["width"]

TRAIN_SUBSET = "TRAIN"
VALIDATE_SUBSET = "VALIDATE"  

AUTOTUNE = tensorflow.data.AUTOTUNE # Automatically Sets the Hyperparameters


def start():
   
    url = URL
    archive = tensorflow.keras.utils.get_file(origin = url, extract = True)
    directory = pathlib.Path(archive).with_suffix('')

    element_count = len(list(directory.glob('*/*.jpg')))

    datasets = tensorflow.data.Dataset.list_files(str(directory/'*/*'), shuffle = False)
    datasets =  datasets.shuffle(element_count, reshuffle_each_iteration = False)

    classes = numpy.array(sorted([item.name for item in directory.glob('*') if item.name != "LICENSE.txt"]))

    validate_size = int(element_count * SPLIT)
    train = datasets.skip(validate_size)
    validate = datasets.take(validate_size)
    
    def get_label(directory):
        # Convert the path to a list of path components
        parts = tensorflow.strings.split(directory, os.path.sep)
        # The second to last is the class-directory
        one_hot = parts[-2] == classes
        # Integer encode the label
        return tensorflow.argmax(one_hot)
    
    def decode_img(image):
        # Convert the compressed string to a 3D uint8 tensor
        image = tensorflow.io.decode_jpeg(image, channels = 3)
        # Resize the image to the desired size
        return tensorflow.image.resize(image, [IMG_HEIGHT, IMG_WEIGHT])
    
    def process_path(directory):
        label = get_label(directory)
        # Load the raw data from the file as a string
        image = tensorflow.io.read_file(directory)
        image = decode_img(image)
        return image, label
    
    train = train.map(process_path, num_parallel_calls = AUTOTUNE)
    validate = validate.map(process_path, num_parallel_calls = AUTOTUNE)

    def configure_for_performance(dataset):
        dataset = dataset.cache()
        dataset = dataset.shuffle(buffer_size = 1000)
        dataset = dataset.batch(BATCH_SIZE)
        dataset = dataset.prefetch(buffer_size = AUTOTUNE)
        return dataset

    # CACHES AND PREFETCH DATA TO MAKE PROCESS MORE EFFICIENT
    train = configure_for_performance(train)
    validate = configure_for_performance(validate)

    dataset = Dataset(classes = classes, train = train, validate = validate)
    
    return dataset

































