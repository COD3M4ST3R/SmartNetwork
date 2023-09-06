
# LOCAL IMPORTS
import AI.Start as AI
from REPOSITORY.Repository import repository 
import CONTROL
import PREDICT.Image
import PLOT.Draw as PLOT

# IMPORTS
import tensorflow
import numpy
import datetime
import subprocess
import sys
import threading
import PySimpleGUI as sg
import os.path
from anyio import sleep


FILTER = repository.HYPERPARAMETER["convolution_layer"]["filter"]
KERNEL = repository.HYPERPARAMETER["convolution_layer"]["kernel"]
SCALE = repository.HYPERPARAMETER["scale"]
NEURONS = repository.HYPERPARAMETER["dense"]["neurons"]
BATCH_SIZE = repository.HYPERPARAMETER["batch"]["size"]
DROPOUT = repository.HYPERPARAMETER["dropout"]

EPOCH = repository.HYPERPARAMETER["epoch"]
DEFAULT = repository.HYPERPARAMETER["learning_rate"]["default"]
DYNAMIC = repository.HYPERPARAMETER["learning_rate"]["dynamic"]
INITIAL_LEARNING_RATE = repository.HYPERPARAMETER["learning_rate"]["initial_learning_rate"]
DECAY_STEPS = repository.HYPERPARAMETER["learning_rate"]["decay_steps"]
DECAY_RATE = repository.HYPERPARAMETER["learning_rate"]["decay_rate"]
STAIRCASE = repository.HYPERPARAMETER["learning_rate"]["staircase"]

OPTIMIZER_ID = repository.OPTIMIZER["id"]
OPTIMIZER_NAME = repository.OPTIMIZER["name"]
OPTIMIZER_BETA_1 = repository.OPTIMIZER["parameters"]["beta_1"]
OPTIMIZER_BETA_2 = repository.OPTIMIZER["parameters"]["beta_2"]
OPTIMIZER_EPSILON = repository.OPTIMIZER["parameters"]["epsilon"]
OPTIMIZER_AMSGRAD = repository.OPTIMIZER["parameters"]["amsgrad"]

OPTIMIZER_weight_decay = repository.OPTIMIZER["parameters"]["weight_decay"]
OPTIMIZER_clipnorm = repository.OPTIMIZER["parameters"]["clipnorm"]
OPTIMIZER_CLIPVALUE = repository.OPTIMIZER["parameters"]["clipvalue"]
OPTIMIZER_GLOBAL_CLIPNORM = repository.OPTIMIZER["parameters"]["global_clipnorm"]
OPTIMIZER_USE_EMA = repository.OPTIMIZER["parameters"]["use_ema"]
OPTIMIZER_USE_MOMENTUM = repository.OPTIMIZER["parameters"]["ema_momentum"]
OPTIMIZER_EMA_OVERWRITE_FREQUENCY = repository.OPTIMIZER["parameters"]["ema_overwrite_frequency"]
OPTIMIZER_JIT_COMPILE = repository.OPTIMIZER["parameters"]["jit_compile"]

PROJECT_TYPE = repository.PROJECT["type"]
PROJECT_TARGET = repository.PROJECT["target"]
PROJECT_DESCRIPTION = repository.PROJECT["description"]

LAP_ID = repository.LAP["id"]
LAP_NAME = repository.LAP["name"]
LAP_DESCRIPTION = repository.LAP["description"]

MODEL_ID = repository.MODEL["id"]
MODEL_NAME = repository.MODEL["name"]
MODEL_DESCRIPTION = repository.MODEL["description"]

ACTIVATION_FUNCTION_NAME = repository.ACTIVATION_FUNCTION["name"]

PREDICT_IMAGE = repository.DATASET["predict"]["url"]

process = "Status: ready."
result = "waiting for training to be completed."


def apply(): 

    def train(window):

        process = "Status: training..."
        window['process'].update(process)

        # LAP (LOAD AND PREPROCESS INPUT)
        dataset = CONTROL.determineLAP.apply()


        # INITIALIZE MODEL
        model = CONTROL.determineModel.apply(num_classes = len(dataset.classes))


        # LEARNING CURVE
        learning_rate = tensorflow.keras.optimizers.schedules.InverseTimeDecay(
            initial_learning_rate = INITIAL_LEARNING_RATE,
            decay_steps = DECAY_STEPS,
            decay_rate = DECAY_RATE,
            staircase = STAIRCASE
        ) if DYNAMIC else DEFAULT


        # COMPILE MODEL
        model.compile(
            optimizer = CONTROL.determineOptimizer.apply(learning_rate = learning_rate),
            loss = tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
            metrics = ['accuracy']
        )


        # FIT MODEL
        history = model.fit(
            dataset.train, 
            validation_data = dataset.validate, 
            epochs = repository.HYPERPARAMETER["epoch"]
        )


        # METRICS
        loss_train = numpy.array(history.history['loss'])
        loss_validate = numpy.array(history.history['val_loss'])
        epochs = numpy.arange(1, len(loss_validate) + 1)
        accuracy_train = history.history['accuracy']
        accuracy_validate = history.history['val_accuracy'] 
        step = numpy.linspace(0, EPOCH)
        running = False


        # PLOT LOSSES
        PLOT.Draw(stamp = True).losses(
            loss_train = loss_train,
            loss_validate = loss_validate,
            epochs = epochs
        )


        # PLOT ACCURACY
        PLOT.Draw(stamp = True).accuracy(
            epochs = epochs,
            accuracy_train = accuracy_train,
            accuracy_validate = accuracy_validate
        )


        # PREDICT
        result = PREDICT.Image.fromDisk(model, dataset)
        window['result'].update(result)

        process = "Status: completed."
        window['process'].update(process)


    theme_dict = {'BACKGROUND': '#2B475D',
                'TEXT': '#FFFFFF',
                'INPUT': '#F2EFE8',
                'TEXT_INPUT': '#000000',
                'SCROLL': '#F2EFE8',
                'BUTTON': ('#000000', '#C2D4D8'),
                'PROGRESS': ('#FFFFFF', '#C7D5E0'),
                'BORDER': 0,'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0}

    sg.theme_add_new('Dashboard', theme_dict)
    sg.theme('Dashboard')

    BORDER_COLOR = '#C7D5E0'
    DARK_HEADER_COLOR = '#1B2838'
    BPAD_TOP = ((20, 20), (20, 10))
    PAD_ELEMENT = ((5, 5), (5, 5))
    PAD_FRAME = ((10, 10), (10, 10))


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


    layout = [
        
        # ROW 1
        [
            sg.Frame('', 
            [
                # ROW 1
                [
                    sg.Text('V0.0 Dashboard', background_color = DARK_HEADER_COLOR, enable_events = True, grab = False), sg.Push(background_color = DARK_HEADER_COLOR),
                    sg.Text('SMART NETWORK', background_color = DARK_HEADER_COLOR),
                    sg.Button('Exit', button_color = ('white', sg.theme_background_color()), border_width = 0, pad = ((20, 0), (0, 0))),
                ],

            ], pad = PAD_FRAME, background_color = DARK_HEADER_COLOR,  expand_x = True, border_width = 0, grab = True),
        ],

        # ROW 2
        [
            sg.Frame('', 
            [
                # ROW 1
                [
                    sg.Push(), sg.Text(f"{PROJECT_TYPE}", font='Any 20'), sg.Push(),
                ],

                # ROW 2
                [
                    sg.T(f"Target: {PROJECT_TARGET}"),
                ],

                # ROW 3
                [
                    sg.T(f"Descripton: {PROJECT_DESCRIPTION}"),
                ],

            ], size=(920, 100), pad = PAD_FRAME,  expand_x=True,  relief=sg.RELIEF_GROOVE, border_width=3),
        ],
        
        # ROW 3
        [
            # COLUMN 1
            sg.Frame('',
            [
                # ROW 1
                [
                    sg.Push(), sg.Text('L.A.P.', font='Any 20'), sg.Push(),
                ],

                # ROW 2
                [
                    sg.T(f"ID: {LAP_ID}"),
                ],

                # ROW 3
                [
                    sg.T(f"Name: {LAP_NAME}"),
                ],

                # ROW 4
                [
                    sg.T(f"Description: {LAP_DESCRIPTION}")
                ],
                
            ], size=(500, 200), pad = PAD_FRAME,  expand_x=True,  relief=sg.RELIEF_GROOVE, border_width=3),

            # COLUMN 2
            sg.Frame('',
            [
                # ROW 1
                [
                    sg.Push(), sg.Text('Model', font='Any 20'), sg.Push(),
                ],

                # ROW 2
                [
                    sg.T(f"ID: {MODEL_ID}"),
                ],

                # ROW 3
                [
                    sg.T(f"Name: {MODEL_NAME}"),
                ],

                # ROW 4
                [
                    sg.T(f"Description: {MODEL_DESCRIPTION}"),
                ],
                
            ], size=(500, 200), pad = PAD_FRAME,  expand_x=True,  relief=sg.RELIEF_GROOVE, border_width=3),
        ], 

        # ROW 4
        [
            # COLUMN 1
            sg.Frame('',
            [
                # ROW 1
                [
                    sg.Push(), sg.Text('Hyperparameters', font='Any 20'), sg.Push(),
                ],

                # ROW 2
                [
                    sg.T(f"Convolution Layer Filter: {FILTER}"),

                ],

                # ROW 3
                [
                    sg.T(f"Convolution Layer Kernel: {KERNEL}"),
                ],

                # ROW 4
                [
                    sg.T(f"Scale: {SCALE}"),
                ],

                # ROW 5
                [
                    sg.T(f"Neurons: {NEURONS}"),
                ],

                # ROW 6
                [
                    sg.T(f"Batch Size: {BATCH_SIZE}"),
                ],

                # ROW 7
                [
                    sg.T(f"Dropout: {DROPOUT}"),
                ],

                # ROW 8
                [
                    sg.T(f"Initial Learning Rate: {INITIAL_LEARNING_RATE}"),
                ],

                # ROW 9
                [
                    sg.T(f"Decay Rate: {DECAY_RATE}"),
                ],

                # ROW 10
                [
                    sg.T(f"Epoch: {EPOCH}"),
                ],

                # ROW 11
                [
                    sg.T(f"Staircase: {STAIRCASE}"),
                ],
                
            ], size=(500, 400), pad = PAD_FRAME,  expand_x=True,  relief=sg.RELIEF_GROOVE, border_width=3),

            # COLUMN 2
            sg.Frame('',
            [
                # ROW 1
                [
                    sg.Push(), sg.Text('Optimizers', font='Any 20'), sg.Push(),
                ],

                # ROW 2
                [
                    sg.T(f"ID: {OPTIMIZER_ID}"),
                ],

                # ROW 4
                [
                    sg.T(f"Name: {OPTIMIZER_NAME}"),                
                ],

                # ROW 5
                [
                    sg.T(f"Beta_1: {OPTIMIZER_BETA_1}"),                   
                ],

                # ROW 6
                [
                    sg.T(f"Beta_2: {OPTIMIZER_BETA_2}"),   
                ],

                # ROW 7
                [
                    sg.T(f"Epsilon: {OPTIMIZER_EPSILON}"),   
                ],

                # ROW 8
                [
                    sg.T(f"Amsgrad: {OPTIMIZER_AMSGRAD}"),   
                ],

                # ROW 9
                [
                    sg.T(f"Weight Decay: {OPTIMIZER_weight_decay}"),   
                ],

                # ROW 10
                [
                    sg.T(f"Clipnorm: {OPTIMIZER_clipnorm}"),   
                ],

                # ROW 11
                [
                    sg.T(f"Clipvalue: {OPTIMIZER_CLIPVALUE}"),   
                ],

                # ROW 12
                [
                    sg.T(f"Global Clipnorm: {OPTIMIZER_GLOBAL_CLIPNORM}"),   
                ],

                # ROW 13
                [
                    sg.T(f"Ema: {OPTIMIZER_USE_EMA}"),   
                ],

                # ROW 14
                [
                    sg.T(f"Momentum: {OPTIMIZER_USE_MOMENTUM}"),   
                ],
                                
                # ROW 15
                [
                    sg.T(f"Ema Overwrite Frequency: {OPTIMIZER_EMA_OVERWRITE_FREQUENCY}"),   
                ],

                # ROW 16
                [
                    sg.T(f"Jit Compile: {OPTIMIZER_JIT_COMPILE}"),   
                ],
                
            ], size=(500, 400), pad = PAD_FRAME,  expand_x=True,  relief=sg.RELIEF_GROOVE, border_width=3),
        ], 

        # ROW 5
        [
            sg.Frame('', 
            [
                # ROW 1
                [
                    sg.T(f"Predict: {PREDICT_IMAGE}"),
                ],

            ], pad = PAD_FRAME, background_color=DARK_HEADER_COLOR,  expand_x=True, border_width=0, grab=True),
        ],

        # ROW 6
        [
            sg.Frame('', 
            [
                # ROW 1
                [
                    sg.T(f"{process}", key = "process"),
                ],

            ], pad = PAD_FRAME, background_color=DARK_HEADER_COLOR,  expand_x=True, border_width=0, grab=True),
        ],

        # ROW 7
        [
            sg.Frame('', 
            [
                # ROW 1
                [
                    sg.T(f"Result: {result}", key = "result"),
                ],

            ], pad = PAD_FRAME, background_color=DARK_HEADER_COLOR,  expand_x=True, border_width=0, grab=True),
        ],

        # ROW 8
        [
            sg.Frame('', 
            [
                # ROW 1
                [
                    sg.Button('Train', size = 150)
                ],

            ], pad = PAD_FRAME, background_color=DARK_HEADER_COLOR,  expand_x=True, border_width=0, grab=True),
        ],

        # ROW LAST
        [
            sg.Sizegrip(background_color=BORDER_COLOR)
        ],
    ]


    window = sg.Window('Dashboard PySimpleGUI-Style', layout, margins=(0,0), background_color=BORDER_COLOR, no_titlebar=True, resizable=True, right_click_menu=sg.MENU_RIGHT_CLICK_EDITME_VER_LOC_EXIT)
    

    running = False
    # Run the Event Loop
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        
        if event == 'Train' and not running:  
            running = True
            threading.Thread(target = train, args=(window,), daemon=True).start()

        if event == "Event":
            print(window.size)


    window.close()
