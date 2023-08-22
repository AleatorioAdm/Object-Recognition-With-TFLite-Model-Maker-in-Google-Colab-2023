"""
Script: Gerado de modelo para reconhecimento de objetos
Descrição: Esse script serve para gerar um modelo de reconhecimento de objetos no formato tflite.
E foi desenvolvido para atuar conforme o código e ambientes de trabalho no link abaixo.
https://colab.research.google.com/drive/1q2EA7AOEdX3KqzYzGU0UO1Ih7tKXMHQd?usp=sharing
Autor: Ademilson Lima da Silva Filho
Data de Criação: 27 de Abril de 2023
Versão: 1.0
Licença: MIT License

Requisitos:
- Python exatamente na versão 3.9
- Biblioteca Tflite Model Maker (instalada com 'pip install tflite-model-maker')
- Biblioteca Pycoco Tools(instalada com 'pip3 install pycocotools')
- Biblioteca Ipykernel (instalada com 'pip install ipykernel')
- Biblioteca Numpy versão 1.23.4 (instalada com 'pip install numpy==1.23.4')
- Biblioteca Torch (instalada com 'pip install torch')
- Biblioteca Torchvision (instalada com 'pip install torchvision')

Notas de Uso:
- Certifique-se de ter as bibliotecas necessárias instaladas antes de executar este script.
- Lembrando que ela foi desenvolvida para executar num ambiente colab e num projeto específico,
ou seja, a depender de seu uso podem ser necessárias modificações no script.
"""

#Primeiro importamos todas as bibliotecas necessárias para a execução do código

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

#Depois definimos a variável que indica o caminho do banco de imagens no Google Colab
image_path = '/content/objects-image-repository'

#Então separamos em pastas para treino e para teste o banco de imagens
train_data_folder, test_data_folder = ImageClassifierDataLoader.from_folder(image_path).split(0.8)

print(len(train_data_folder))

print(len(test_data_folder))

#Logo, podemos criar, treinar e validar o modelo
modelf = image_classifier.create(train_data_folder, epochs=10,validation_data=test_data_folder)


#Fazemos uma avaliação da precisão do modelo
loss, accuracy = modelf.evaluate(test_data_folder)


#Exportamos o modelo para um arquivo tflite
modelf.export(export_dir='.')


#Avaliamos a precisão do modelo após exportado, para garantir que foi exportado corretamente
modelf.evaluate_tflite('model.tflite', test_data_folder)
