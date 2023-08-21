import os

import numpy as np

import tensorflow as tf
assert tf.__version__.startswith('2')

from tflite_model_maker import image_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import torchvision as torch

from tensorflow_examples.lite.model_maker.core.data_util.image_dataloader import ImageClassifierDataLoader

import matplotlib.pyplot as plt
import tensorflow.keras as keras
from tensorflow.keras import datasets, layers, models


image_path = '/content/objects-image-repository'

train_data_folder, test_data_folder = ImageClassifierDataLoader.from_folder(image_path).split(0.8)

print(len(train_data_folder))

print(len(test_data_folder))


modelf = image_classifier.create(train_data_folder, epochs=10,validation_data=test_data_folder)



loss, accuracy = modelf.evaluate(test_data_folder)



modelf.export(export_dir='.')



modelf.evaluate_tflite('model.tflite', test_data_folder)