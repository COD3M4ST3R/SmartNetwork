
# LOCAL IMPORTS
from REPOSITORY.Repository import repository
from LAP import lap_keras, lap_pipeline


def apply():
    id = repository.LAP["id"]
    
    match id:
        case 1:
            # lap_keras
            return lap_keras.start()
        case 2:
            # lap_pipeline
            return lap_pipeline.start()
        case _:
            # use 'case 1' as default.
            return lap_keras.start()
