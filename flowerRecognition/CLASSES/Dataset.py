

class Dataset:
    def __init__(self, classes = None,
                train = None, 
                validate = None):

        self._classes = classes
        self._train = train
        self._validate = validate

    @property
    def classes(self):
        return self._classes

    @property
    def train(self):
        return self._train

    @property
    def validate(self):
        return self._validate
    
