
# LOCAL IMPORTS
from REPOSITORY.Repository import repository
import CONTROL.determineLAP
import CONTROL.determineModel
import CONTROL.determineOptimizer
import PLOT.Draw as PLOT
import PREDICT.Image

# IMPORTS
import tensorflow 
import numpy
from matplotlib import pyplot as plt
import PySimpleGUI as sg
import os.path

# CONSTANTS
AUTOTUNE = tensorflow.data.AUTOTUNE # Automatically Sets the Hyperparameters
EPOCH = repository.HYPERPARAMETER["epoch"]
DEFAULT = repository.HYPERPARAMETER["learning_rate"]["default"]
DYNAMIC = repository.HYPERPARAMETER["learning_rate"]["dynamic"]
INITIAL_LEARNING_RATE = repository.HYPERPARAMETER["learning_rate"]["initial_learning_rate"]
DECAY_STEPS = repository.HYPERPARAMETER["learning_rate"]["decay_steps"]
DECAY_RATE = repository.HYPERPARAMETER["learning_rate"]["decay_rate"]
STAIRCASE = repository.HYPERPARAMETER["learning_rate"]["staircase"]



# # LAP (LOAD AND PREPROCESS INPUT)
# dataset = CONTROL.determineLAP.apply()


# # INITIALIZE MODEL
# model = CONTROL.determineModel.apply(num_classes = len(dataset.classes))


# # LEARNING CURVE
# learning_rate = tensorflow.keras.optimizers.schedules.InverseTimeDecay(
#     initial_learning_rate = INITIAL_LEARNING_RATE,
#     decay_steps = DECAY_STEPS,
#     decay_rate = DECAY_RATE,
#     staircase = STAIRCASE
# ) if DYNAMIC else DEFAULT


# # COMPILE MODEL
# model.compile(
#     optimizer = CONTROL.determineOptimizer.apply(learning_rate = learning_rate),
#     loss = tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
#     metrics = ['accuracy']
# )


# # FIT MODEL
# history = model.fit( # During the model.fit() call, the model is trained using the specified optimizer, loss function, and evaluation metrics. The training process involves forward propagation, backward propagation (gradient calculation), weight updates, and iterative optimization. After each epoch, the model's performance on the validation dataset is evaluated and displayed.
#   dataset.train, # Training dataset has been defined 
#   validation_data = dataset.validate, # The validation dataset used to evaluate the model's performance on unseen data after each training epoch. It helps to monitor the model's generalization and detect overfitting.
#   epochs = repository.HYPERPARAMETER["epoch"] # The number if epoch determines how many times the entire training dataset will be iterated during training. One epoch is defined as a complete pass through the entire training dataset.
# )


# # METRICS
# loss_train = numpy.array(history.history['loss'])
# loss_validate = numpy.array(history.history['val_loss'])
# epochs = numpy.arange(1, len(loss_validate) + 1)
# accuracy_train = history.history['accuracy']
# accuracy_validate = history.history['val_accuracy'] 
# step = numpy.linspace(0, EPOCH)


# # PLOT LOSSES
# PLOT.Draw(stamp = True).losses(
#     loss_train = loss_train,
#     loss_validate = loss_validate,
#     epochs = epochs
# )


# # PLOT ACCURACY
# PLOT.Draw(stamp = True).accuracy(
#     epochs = epochs,
#     accuracy_train = accuracy_train,
#     accuracy_validate = accuracy_validate
# )


# # PREDICT
# PREDICT.Image.fromDisk(model, dataset)


# UI
column_attributes = [
    [sg.Text(f"Pipeline: {repository.LAP['function_name']}", justification = 'center')],
    [sg.Text(f"Activation Function: {repository.ACTIVATION_FUNCTION['name']}", justification = 'center')],
    [sg.Text(f"Optimizer: {repository.OPTIMIZER['name']}", justification = 'center')],
    [sg.Text(f"Model: {repository.MODEL['name']}", justification = 'center')],

]


column_fileNames = [
    [
        sg.Text("Image Folder"),
        sg.In(size = (25, 1), enable_events = True, key="-FOLDER-"),
        sg.FolderBrowse(),
    ],
    [
        sg.Listbox(
            values = [], enable_events = True, size = (40, 20), key = "-FILE LIST-"
        )
    ],
]

# For now will only show the name of the file that was chosen
column_imageView = [
    [sg.Text("Choose an image from list on left:")],
    [sg.Text(size=(40, 1), key="-TOUT-")],
    [sg.Image(key="-IMAGE-")],
]

# ----- Full layout -----
layout = [
    [
        sg.Column(column_attributes),
        sg.HSeparator(),
        sg.Column(column_fileNames),
        sg.VSeperator(),
        sg.Column(column_imageView),
    ]
]

window = sg.Window("PREDICT", layout)

# Run the Event Loop
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    # Folder name was filled in, make a list of files in the folder
    if event == "-FOLDER-":
        folder = values["-FOLDER-"]
        try:
            # Get list of files in folder
            file_list = os.listdir(folder)
        except:
            file_list = []

        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder, f))
            and f.lower().endswith((".*"))
        ]
        window["-FILE LIST-"].update(fnames)
    elif event == "-FILE LIST-":  # A file was chosen from the listbox
        try:
            filename = os.path.join(
                values["-FOLDER-"], values["-FILE LIST-"][0]
            )
            window["-TOUT-"].update(filename)
            window["-IMAGE-"].update(filename=filename)

        except:
            pass

window.close()

