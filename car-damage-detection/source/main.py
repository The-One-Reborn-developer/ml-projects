from kagglehub import dataset_download

from pathlib import Path

from fastai.vision.all import (
    ImageDataLoaders,
    Resize,
    aug_transforms,
    vision_learner,
    error_rate,
    resnet34
)


path_string = dataset_download("anujms/car-damage-detection")
path = Path(path_string) / 'data1a'

data_block = ImageDataLoaders.from_folder(
    path,
    'training',
    'validation',
    seed=42,
    item_tfms=Resize(400),
    batch_tfms=aug_transforms()
)

learner = vision_learner(data_block, resnet34, metrics=error_rate)
learner.fine_tune(1)

test = Path('.') / 'test' / '1.jpg'
learner.predict(test)
