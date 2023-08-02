
class Lap:
    def __init__(self, 
                 id = None, 
                 name = None, 
                 function_name = None, 
                 description = None, 
                 active = None):
        self._id = id
        self._name = name
        self._function_name = function_name
        self._description = description
        self._active = active

    @property   
    def id(self):
        return self._id
    
    @property
    def name(self):
        return self._name
    
    @property
    def name(self):
        return self._function_name
    
    @property
    def description(self):
        return self._description
    
    @property
    def active(self):
        return self._active
    