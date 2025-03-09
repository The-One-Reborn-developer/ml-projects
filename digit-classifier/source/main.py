import pandas

from IPython.display import display
from PIL import Image
from pathlib import Path

from fastai.vision.all import (
    URLs,
    untar_data,
    tensor,
    show_image
)


dataset_path_string = untar_data(URLs.MNIST)
dataset_path = Path(dataset_path_string) / 'training'

zeroes = (dataset_path / '0').ls().sorted()
ones = (dataset_path / '1').ls().sorted()
twos = (dataset_path / '2').ls().sorted()
threes = (dataset_path / '3').ls().sorted()
fours = (dataset_path / '4').ls().sorted()
fives = (dataset_path / '5').ls().sorted()
sixs = (dataset_path / '6').ls().sorted()
sevens = (dataset_path / '7').ls().sorted()
eights = (dataset_path / '8').ls().sorted()
nines = (dataset_path / '9').ls().sorted()

zeroes_tensors = [tensor(Image.open(image)) for image in zeroes]
ones_tensors = [tensor(Image.open(image)) for image in ones]
twos_tensors = [tensor(Image.open(image)) for image in twos]
threes_tensors = [tensor(Image.open(image)) for image in threes]
fours_tensors = [tensor(Image.open(image)) for image in fours]
fives_tensors = [tensor(Image.open(image)) for image in fives]
sixs_tensors = [tensor(Image.open(image)) for image in sixs]
sevens_tensors = [tensor(Image.open(image)) for image in sevens]
eights_tensors = [tensor(Image.open(image)) for image in eights]
nines_tensors = [tensor(Image.open(image)) for image in nines]
