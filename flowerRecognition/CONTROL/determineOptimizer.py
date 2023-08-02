
# LOCAL IMPORTS
from REPOSITORY.Repository import repository

# IMPORTS
import tensorflow


def apply(learning_rate):
    try:
        optimizer = repository.OPTIMIZER
        id = optimizer['id']
    except:
        if id is None:
            print('There is no any active optimizer that has been selected. Using default optimizer now...')
            id = 1
        else:
            print('Error occured during selection of optimizer. Using default optimizer now...')

    match id:
        case 1:
            return tensorflow.keras.optimizers.Adam(optimizer = optimizer, learning_rate = learning_rate)
        case 2:
            return tensorflow.keras.optimizers.Adadelta(optimizer = optimizer, learning_rate = learning_rate)
        case 3:
            return tensorflow.keras.optimizers.Adafactor(optimizer = optimizer, learning_rate = learning_rate)
        case 4:
            return tensorflow.keras.optimizers.Adamax(optimizer = optimizer, learning_rate = learning_rate)
        case 5: 
            return tensorflow.keras.optimizers.Adagrad(optimizer = optimizer, learning_rate = learning_rate)
        case 6:
            return tensorflow.keras.optimizers.AdamW(optimizer = optimizer, learning_rate = learning_rate)
        case 7:
            return tensorflow.keras.optimizers.Nadam(optimizer = optimizer, learning_rate = learning_rate)
        case _:
            # use 'case 1' as default.
            return tensorflow.optimizers.Adam(optimizer = optimizer, learning_rate = learning_rate)
