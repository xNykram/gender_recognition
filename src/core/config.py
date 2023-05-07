from dataclasses import dataclass
from os import environ

@dataclass
class Settings():
    NUM_EPOCH: int = int(environ.get('NUM_EPOCH', 0))
    PATH_TRANING_IMAGES: str = 'src/data/raw/images/traning_images'
    PATH_PREDICT_IMAGES: str = 'src/data/raw/images/prediction_images'

settings = Settings()
