
'''
This file reads the parameter file in the path of 'PARAMETER_PATH'to initialize repository with all the necessary
values where you can access them across anywhere in this project.
'''

# IMPORTS
import ruamel.yaml

# CONSTANTS
PARAMETER_PATH = "PARAMETERS/values.yaml"


class Repository:
    def __init__(self):
        self.data = self.load_yaml()


    def load_yaml(self):
        with open(PARAMETER_PATH, 'r') as file:
            yaml = ruamel.yaml.YAML(typ='safe')
            data = yaml.load(file)
        
        return data


    def __getattr__(self, item):
        return self._get_nested_value(item)


    def _get_nested_value(self, nested_key):
        keys = nested_key.split('.')
        value = self.data
        for key in keys:
            if isinstance(value, list):
                value = [item.get(key) for item in value]
            else:
                value = value.get(key)
            if value is None:
                break

        return value


    def __getitem__(self, item):
        return self._get_nested_value(item)
    

    @property
    def LAP(self):
        lap_items = self.data.get('LAP', [])
        active_lap = next((item for item in lap_items if item.get('active', True)), None)

        return active_lap
    

    @property
    def MODEL(self):
        model_items = self.data.get('MODEL', [])
        active_model = next((item for item in model_items if item.get('active', True)), None)

        return active_model
    

    @property
    def OPTIMIZER(self):
        optimizer_items = self.data.get('OPTIMIZER', [])
        active_optimizer = next((item for item in optimizer_items if item.get('active', True)), None)

        return active_optimizer
    

repository = Repository()

