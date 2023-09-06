
# LOCAL IMPORTS
from REPOSITORY.Repository import repository

# IMPORTS
import tensorflow
import numpy

# CONSTANTS
URL = repository.DATASET["predict"]["url"]
IMG_HEIGHT = repository.INPUT["image"]["dimension"]["height"]
IMG_WIDTH = repository.INPUT["image"]["dimension"]["width"]



def fromURL(model, dataset):
    target_url = tensorflow.keras.utils.get_file(origin = URL)

    target_image = tensorflow.keras.utils.load_img(target_url, target_size = (IMG_HEIGHT, IMG_WIDTH))
    
    images = tensorflow.keras.utils.img_to_array(target_image)
    images = tensorflow.expand_dims(images, 0) # Create a batch
    images = images / 255.0  # Normalize the pixel values between 0 and 1

    predictions = model.predict(images)
    score = tensorflow.nn.softmax(predictions[0])

    result = "This image most likely belongs to {} with a {:.2f} percent confidence.".format(dataset.classes[numpy.argmax(score)], 100 * numpy.max(score))

    return result


def fromDisk(model, dataset):

    target_image = tensorflow.keras.utils.load_img(URL, target_size = (IMG_HEIGHT, IMG_WIDTH))
    
    images = tensorflow.keras.utils.img_to_array(target_image)
    images = tensorflow.expand_dims(images, 0) # Create a batch
    images = images / 255.0  # Normalize the pixel values between 0 and 1

    predictions = model.predict(images)
    score = tensorflow.nn.softmax(predictions[0])

    result = "This image most likely belongs to {} with a {:.2f} percent confidence.".format(dataset.classes[numpy.argmax(score)], 100 * numpy.max(score))

    return result