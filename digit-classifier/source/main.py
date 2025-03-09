import pandas

from IPython.display import display
from PIL import Image
from pathlib import Path

from fastai.vision.all import (
    URLs,
    untar_data,
    vision_learner,
    tensor,
    array
)


dataset_path_string = untar_data(URLs.MNIST)
dataset_path = Path(dataset_path_string) / 'training'

zeroes = (dataset_path / '0').ls().sorted()
ones = (dataset_path / '1').ls().sorted()

image_zero = zeroes[0]
image = Image.open(image_zero)
image_tensor = tensor(image)

data_frame = pandas.DataFrame(image_tensor)
display(data_frame)