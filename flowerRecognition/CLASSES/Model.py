
class Model:
    def __init__(self,
                 id = None,
                 active = None,
                 name = None,
                 scale = None,
                 output_channel = None,
                 convolution_kernel = None):
        self._id = id,
        self._active = active   
        self._name = name
        self._scale = scale
        self._output_channel = output_channel
        self._convolution_kernel = convolution_kernel

    @property
    def id(self):
        return self._id
    
    @property
    def active(self):
        return self._active
    
    @property
    def name(self):
        return self._name
    
    @property
    def scale(self):
        return self._scale
    
    @property
    def output_channel(self):
        return self._output_channel
    
    @property
    def convolution_kernel(self):
        return self._convolution_kernel
        