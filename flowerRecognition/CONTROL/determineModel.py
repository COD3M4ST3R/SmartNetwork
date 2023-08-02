
# LOCAL IMPORTS
from REPOSITORY.Repository import repository
from MODEL.sequential import sequential


def apply(num_classes: int):
    id = repository.MODEL["id"]  
    
    match id:
        case 1:
            # sequential
            return sequential(num_classes = num_classes)
        case _:
            # use 'case 1' as default.
            return sequential(num_classes = num_classes)
