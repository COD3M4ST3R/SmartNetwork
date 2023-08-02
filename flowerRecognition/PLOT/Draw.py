
# LOCAL IMPORTS
from REPOSITORY.Repository import repository

# IMPORTS
from matplotlib import pyplot as plt
import numpy
from tensorflow.python.framework.ops import EagerTensor


class Draw:
    def __init__(
            self,
            stamp = None,
            **kwargs
    ):
        self._stamp = stamp 

    # Adds Stamp to Figure
    def __stamp(self, fig: plt.figure, plt: plt) -> None: # adds stamp to figure
            fig.text(0.5, 0.015, "Nadir Suhan ILTER | Image Classification", horizontalalignment = "center")
            plt.subplots_adjust(top = 0.8, bottom = 0.2)

    # Draws Losses of the Model While Learning for Training Dataset and Validating Dataset
    def losses(self, loss_train: numpy.ndarray, loss_validate: numpy.ndarray, epochs: numpy.ndarray):
        fig = plt.figure()

        if self._stamp == True:
            self.__stamp(fig = fig, plt = plt)

        fig.text(0.5, 0.9, 'Model: {}\n Neurons: {}\n Optimizer: {}\n'.format(repository.MODEL['name'], repository.HYPERPARAMETER['dense']['neurons'], repository.OPTIMIZER['name']), fontsize = 9, horizontalalignment = "center")
        plt.plot(epochs, loss_train, label = 'Loss during training', linestyle = 'dashed')
        plt.plot(epochs, loss_validate, label = 'Loss during validate')
        plt.title('Sparse Categorical Crossentropy Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def learning_rate(self, step: numpy.ndarray, EPOCH: int, learning_rate: EagerTensor):
        fig = plt.figure()

        if self._stamp == True:
            self.__stamp(fig = fig, plt = plt)

        plt.plot(step, learning_rate, label = 'Learning Rate')
        plt.title('Learning Rate Over Time for Optimizer')
        plt.ylim([0, max(plt.ylim())])  
        plt.xlabel('Epochs')
        plt.ylabel('Rate')
        plt.legend()
        plt.show()

    # Draws Accuracy of the Model on Training Dataset and Validating Dataset
    def accuracy(self, epochs: numpy.ndarray, accuracy_train: list, accuracy_validate: list):
        fig = plt.figure()

        if self._stamp == True:
             self.__stamp(fig = fig, plt = plt)

        plt.plot(epochs, accuracy_train, label = "Accuracy of Training", linestyle = 'dashed')
        plt.plot(epochs, accuracy_validate, label = "Accuracy of Validating")
        plt.title('Accuracies') 
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()


# # PLOTS ON SAME FIGURE
# fig, axis = plt.subplots(1, 2)
# fig.text(0.5, 0.9, 'Model: {}\n Neurons: {}\n Optimizer: {}\n'.format(repository.MODEL['name'], repository.HYPERPARAMETER['dense']['neurons'], repository.OPTIMIZER['name']), fontsize = 9, horizontalalignment = "center")
# fig.text(0.5, 0.015, "Nadir Suhan ILTER | Image Classification", horizontalalignment = "center")
# plt.subplots_adjust(top = 0.8, bottom = 0.2)


# # PLOT LOSSES OF DATASETS
# loss_train = numpy.array(history.history['loss'])
# loss_validate = numpy.array(history.history['val_loss'])
# epochs = numpy.arange(1, len(loss_validate) + 1)

# axis[0].plot(epochs, loss_train, label = 'Loss during training')
# axis[0].plot(epochs, loss_validate, label = 'Loss during validate')
# axis[0].set_title('Sparse Categorical Crossentropy Loss')
# axis[0].set_xlabel('Epochs')
# axis[0].set_ylabel('Loss')
# axis[0].legend()


# # LEARNING RATE PLOT
# step = numpy.linspace(0, 10)
# lr = lr_schedule(step)

# axis[1].plot(step / EPOCH, lr, label = 'Learning Rate Over Time for Optimizer')
# axis[1].set_title('Learning Rate')
# axis[1].set_ylim([0, max(plt.ylim())])  
# axis[1].set_xlabel('Epochs')
# axis[1].set_ylabel('Rate')
# axis[1].legend()


# plt.show()