from numpy import loadtxt

from kagglehub import dataset_download

from pathlib import Path

from fastai.vision.all import (
    SegmentationDataLoaders,
    Resize,
    get_image_files,
    resnet34,
    unet_learner
)


dataset_path_string = dataset_download('intelecai/car-segmentation')
dataset_path = Path(dataset_path_string) / 'car-segmentation'

data_block = SegmentationDataLoaders.from_label_func(
    dataset_path,
    fnames=get_image_files(dataset_path / 'images'),
    label_func=lambda file_name: dataset_path / 'masks' / file_name.name,
    codes=loadtxt(dataset_path / 'classes.txt', dtype=str),
    item_tfms=Resize(224)
)


learner = unet_learner(data_block, resnet34)
learner.fine_tune(8)

learner.show_results(max_n=4)
