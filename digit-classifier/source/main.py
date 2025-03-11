import torch

from PIL import Image
from pathlib import Path

from fastai.vision.all import (
    URLs,
    untar_data,
    tensor
)


def mnist_distance(digit_tensor, ideal_mean_for_that_digit):
    return (ideal_mean_for_that_digit - digit_tensor).abs().mean((-1, -2))


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

zeroes_tensors_list = [tensor(Image.open(image)) for image in zeroes]
ones_tensors_list = [tensor(Image.open(image)) for image in ones]
twos_tensors_list = [tensor(Image.open(image)) for image in twos]
threes_tensors_list = [tensor(Image.open(image)) for image in threes]
fours_tensors_list = [tensor(Image.open(image)) for image in fours]
fives_tensors_list = [tensor(Image.open(image)) for image in fives]
sixs_tensors_list = [tensor(Image.open(image)) for image in sixs]
sevens_tensors_list = [tensor(Image.open(image)) for image in sevens]
eights_tensors_list = [tensor(Image.open(image)) for image in eights]
nines_tensors_list = [tensor(Image.open(image)) for image in nines]

stacked_zeroes = torch.stack(zeroes_tensors_list).float()/255
stacked_ones = torch.stack(ones_tensors_list).float()/255
stacked_twos = torch.stack(twos_tensors_list).float()/255
stacked_threes = torch.stack(threes_tensors_list).float()/255
stacked_fours = torch.stack(fours_tensors_list).float()/255
stacked_fives = torch.stack(fives_tensors_list).float()/255
stacked_sixs = torch.stack(sixs_tensors_list).float()/255
stacked_sevens = torch.stack(sevens_tensors_list).float()/255
stacked_eights = torch.stack(eights_tensors_list).float()/255
stacked_nines = torch.stack(nines_tensors_list).float()/255

zeroes_mean = stacked_zeroes.mean(0)
ones_mean = stacked_ones.mean(0)
twos_mean = stacked_twos.mean(0)
threes_mean = stacked_threes.mean(0)
fours_mean = stacked_fours.mean(0)
fives_mean = stacked_fives.mean(0)
sixs_mean = stacked_sixs.mean(0)
sevens_mean = stacked_sevens.mean(0)
eights_mean = stacked_eights.mean(0)
nines_mean = stacked_nines.mean(0)

validation_dataset_path = Path(dataset_path_string) / 'testing'

validation_zeroes = (validation_dataset_path / '0').ls().sorted()
validation_ones = (validation_dataset_path / '1').ls().sorted()
validation_twos = (validation_dataset_path / '2').ls().sorted()
validation_threes = (validation_dataset_path / '3').ls().sorted()
validation_fours = (validation_dataset_path / '4').ls().sorted()
validation_fives = (validation_dataset_path / '5').ls().sorted()
validation_sixs = (validation_dataset_path / '6').ls().sorted()
validation_sevens = (validation_dataset_path / '7').ls().sorted()
validation_eights = (validation_dataset_path / '8').ls().sorted()
validation_nines = (validation_dataset_path / '9').ls().sorted()

validation_zeroes_tensors_list = [tensor(Image.open(image)) for image in validation_zeroes]
validation_ones_tensors_list = [tensor(Image.open(image)) for image in validation_ones]
validation_twos_tensors_list = [tensor(Image.open(image)) for image in validation_twos]
validation_threes_tensors_list = [tensor(Image.open(image)) for image in validation_threes]
validation_fours_tensors_list = [tensor(Image.open(image)) for image in validation_fours]
validation_fives_tensors_list = [tensor(Image.open(image)) for image in validation_fives]
validation_sixs_tensors_list = [tensor(Image.open(image)) for image in validation_sixs]
validation_sevens_tensors_list = [tensor(Image.open(image)) for image in validation_sevens]
validation_eights_tensors_list = [tensor(Image.open(image)) for image in validation_eights]
validation_nines_tensors_list = [tensor(Image.open(image)) for image in validation_nines]

validation_stacked_zeroes = torch.stack(validation_zeroes_tensors_list).float() / 255
validation_stacked_ones = torch.stack(validation_ones_tensors_list).float() / 255
validation_stacked_twos = torch.stack(validation_twos_tensors_list).float() / 255
validation_stacked_threes = torch.stack(validation_threes_tensors_list).float() / 255
validation_stacked_fours = torch.stack(validation_fours_tensors_list).float() / 255
validation_stacked_fives = torch.stack(validation_fives_tensors_list).float() / 255
validation_stacked_sixs = torch.stack(validation_sixs_tensors_list).float() / 255
validation_stacked_sevens = torch.stack(validation_sevens_tensors_list).float() / 255
validation_stacked_eights = torch.stack(validation_eights_tensors_list).float() / 255
validation_stacked_nines = torch.stack(validation_nines_tensors_list).float() / 255

def is_zero(digit_tensor):
    distances = torch.stack([
        mnist_distance(digit_tensor, zeroes_mean),
        mnist_distance(digit_tensor, ones_mean),
        mnist_distance(digit_tensor, twos_mean),
        mnist_distance(digit_tensor, threes_mean),
        mnist_distance(digit_tensor, fours_mean),
        mnist_distance(digit_tensor, fives_mean),
        mnist_distance(digit_tensor, sixs_mean),
        mnist_distance(digit_tensor, sevens_mean),
        mnist_distance(digit_tensor, eights_mean),
        mnist_distance(digit_tensor, nines_mean),
    ])
    closest_digit = distances.argmin(dim=0)
    return closest_digit == 0

def is_one(digit_tensor):
    distances = torch.stack([
        mnist_distance(digit_tensor, zeroes_mean),
        mnist_distance(digit_tensor, ones_mean),
        mnist_distance(digit_tensor, twos_mean),
        mnist_distance(digit_tensor, threes_mean),
        mnist_distance(digit_tensor, fours_mean),
        mnist_distance(digit_tensor, fives_mean),
        mnist_distance(digit_tensor, sixs_mean),
        mnist_distance(digit_tensor, sevens_mean),
        mnist_distance(digit_tensor, eights_mean),
        mnist_distance(digit_tensor, nines_mean),
    ])
    closest_digit = distances.argmin(dim=0)
    return closest_digit == 1

def is_two(digit_tensor):
    distances = torch.stack([
        mnist_distance(digit_tensor, zeroes_mean),
        mnist_distance(digit_tensor, ones_mean),
        mnist_distance(digit_tensor, twos_mean),
        mnist_distance(digit_tensor, threes_mean),
        mnist_distance(digit_tensor, fours_mean),
        mnist_distance(digit_tensor, fives_mean),
        mnist_distance(digit_tensor, sixs_mean),
        mnist_distance(digit_tensor, sevens_mean),
        mnist_distance(digit_tensor, eights_mean),
        mnist_distance(digit_tensor, nines_mean),
    ])
    closest_digit = distances.argmin(dim=0)
    return closest_digit == 2

def is_three(digit_tensor):
    distances = torch.stack([
        mnist_distance(digit_tensor, zeroes_mean),
        mnist_distance(digit_tensor, ones_mean),
        mnist_distance(digit_tensor, twos_mean),
        mnist_distance(digit_tensor, threes_mean),
        mnist_distance(digit_tensor, fours_mean),
        mnist_distance(digit_tensor, fives_mean),
        mnist_distance(digit_tensor, sixs_mean),
        mnist_distance(digit_tensor, sevens_mean),
        mnist_distance(digit_tensor, eights_mean),
        mnist_distance(digit_tensor, nines_mean),
    ])
    closest_digit = distances.argmin(dim=0)
    return closest_digit == 3

def is_four(digit_tensor):
    distances = torch.stack([
        mnist_distance(digit_tensor, zeroes_mean),
        mnist_distance(digit_tensor, ones_mean),
        mnist_distance(digit_tensor, twos_mean),
        mnist_distance(digit_tensor, threes_mean),
        mnist_distance(digit_tensor, fours_mean),
        mnist_distance(digit_tensor, fives_mean),
        mnist_distance(digit_tensor, sixs_mean),
        mnist_distance(digit_tensor, sevens_mean),
        mnist_distance(digit_tensor, eights_mean),
        mnist_distance(digit_tensor, nines_mean),
    ])
    closest_digit = distances.argmin(dim=0)
    return closest_digit == 4

def is_five(digit_tensor):
    distances = torch.stack([
        mnist_distance(digit_tensor, zeroes_mean),
        mnist_distance(digit_tensor, ones_mean),
        mnist_distance(digit_tensor, twos_mean),
        mnist_distance(digit_tensor, threes_mean),
        mnist_distance(digit_tensor, fours_mean),
        mnist_distance(digit_tensor, fives_mean),
        mnist_distance(digit_tensor, sixs_mean),
        mnist_distance(digit_tensor, sevens_mean),
        mnist_distance(digit_tensor, eights_mean),
        mnist_distance(digit_tensor, nines_mean),
    ])
    closest_digit = distances.argmin(dim=0)
    return closest_digit == 5

def is_six(digit_tensor):
    distances = torch.stack([
        mnist_distance(digit_tensor, zeroes_mean),
        mnist_distance(digit_tensor, ones_mean),
        mnist_distance(digit_tensor, twos_mean),
        mnist_distance(digit_tensor, threes_mean),
        mnist_distance(digit_tensor, fours_mean),
        mnist_distance(digit_tensor, fives_mean),
        mnist_distance(digit_tensor, sixs_mean),
        mnist_distance(digit_tensor, sevens_mean),
        mnist_distance(digit_tensor, eights_mean),
        mnist_distance(digit_tensor, nines_mean),
    ])
    closest_digit = distances.argmin(dim=0)
    return closest_digit == 6

def is_seven(digit_tensor):
    distances = torch.stack([
        mnist_distance(digit_tensor, zeroes_mean),
        mnist_distance(digit_tensor, ones_mean),
        mnist_distance(digit_tensor, twos_mean),
        mnist_distance(digit_tensor, threes_mean),
        mnist_distance(digit_tensor, fours_mean),
        mnist_distance(digit_tensor, fives_mean),
        mnist_distance(digit_tensor, sixs_mean),
        mnist_distance(digit_tensor, sevens_mean),
        mnist_distance(digit_tensor, eights_mean),
        mnist_distance(digit_tensor, nines_mean),
    ])
    closest_digit = distances.argmin(dim=0)
    return closest_digit == 7

def is_eight(digit_tensor):
    distances = torch.stack([
        mnist_distance(digit_tensor, zeroes_mean),
        mnist_distance(digit_tensor, ones_mean),
        mnist_distance(digit_tensor, twos_mean),
        mnist_distance(digit_tensor, threes_mean),
        mnist_distance(digit_tensor, fours_mean),
        mnist_distance(digit_tensor, fives_mean),
        mnist_distance(digit_tensor, sixs_mean),
        mnist_distance(digit_tensor, sevens_mean),
        mnist_distance(digit_tensor, eights_mean),
        mnist_distance(digit_tensor, nines_mean),
    ])
    closest_digit = distances.argmin(dim=0)
    return closest_digit == 8

def is_nine(digit_tensor):
    distances = torch.stack([
        mnist_distance(digit_tensor, zeroes_mean),
        mnist_distance(digit_tensor, ones_mean),
        mnist_distance(digit_tensor, twos_mean),
        mnist_distance(digit_tensor, threes_mean),
        mnist_distance(digit_tensor, fours_mean),
        mnist_distance(digit_tensor, fives_mean),
        mnist_distance(digit_tensor, sixs_mean),
        mnist_distance(digit_tensor, sevens_mean),
        mnist_distance(digit_tensor, eights_mean),
        mnist_distance(digit_tensor, nines_mean),
    ])
    closest_digit = distances.argmin(dim=0)
    return closest_digit == 9

accuracy_zeroes = is_zero(validation_stacked_zeroes).float().mean()
accuracy_ones = is_one(validation_stacked_ones).float().mean()
accuracy_twos = is_two(validation_stacked_twos).float().mean()
accuracy_threes = is_three(validation_stacked_threes).float().mean()
accuracy_fours = is_four(validation_stacked_fours).float().mean()
accuracy_fives = is_five(validation_stacked_fives).float().mean()
accuracy_sixs = is_six(validation_stacked_sixs).float().mean()
accuracy_sevens = is_seven(validation_stacked_sevens).float().mean()
accuracy_eights = is_eight(validation_stacked_eights).float().mean()
accuracy_nines = is_nine(validation_stacked_nines).float().mean()
