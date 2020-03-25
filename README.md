# ModeLIB - SoTA Classification models in Keras / Pytorch
[![License: WTFPL](https://img.shields.io/badge/License-WTFPL-lightgrey.svg)](http://www.wtfpl.net/about/)
![build](https://img.shields.io/badge/build-unstable-orange.svg)

## Overview
This repository contains reimplementation of state-of-the-art **Image Classification** models:

| **Models**| **Published year**| **Paper**  |
|------|-------------| -----|
| VGG16|2014| [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) |
| InceptionV3|2015| [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567) |
| ResNet50 |2015|[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)|
|DenseNet121|2016|[Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)|
|Xception|2016|[Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)|
|ResNeXt50|2016|[Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)|
|MobileNetV3|2019|[Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)|



The goal of this implementation is to be simple, flat structured,  
highly extensible, and easy to integrate into your own projects.

**At current moment, you can easily:**  
 * Build any of the above models in nn.Module class 
 * Use models for classification or feature extraction 
 * Train/Test models with your in-house data
 * Evaluate models by validation dataset within training

**_Upcoming features_: In the next few days, you will be able to:**
 * Local installation / pip installation
 
**Pre-trained model will update ASAP, we're seeking for proper cloud storage now.**
 * Load pre-trained weights for models  
 * Finetune models on your own dataset

---
### Table of contents
1. [About Models](#about-models)
    * [Model Details](#model-details)
2. [Requesties](#requesties)
3. [Installation](#installation)
4. [Usage](#usage)
    * [Example: Train from bash](#example-train-from-bash)
    * [Example: Customize in python](#example-customize-in-python)
5. [Contributing](#contributing)
6. [LICENSE](#liscence)

---
## About Models

We collect models that bring huge impacts to Image Classification task,
 and re-implement it with flat structured code. 
Although most of these models have several types where their layers
 altered such as ResNet18/30/50, we only implement one of those in order to
 remain code legibility.

### Model Details
All Params are calculated with input size=(224, 224, 3) and classes=1000.
Top-1 Acc. were evaluated by pre-trained model on ImageNet dataset

|*Name*| `keras`| `torch` |*Pretrained*|*# Params*|*Top-1 Acc.*|
|:---:|:--------:|:---:|---:|------:|:-----------:|
| VGG16|v|v|x|138,357,544|?|
| InceptionV3|v|v|x|25,081,224|?|
| ResNet50 |v|v|x|25,621,352|?|
|DenseNet121|v|v|x|7,899,430|?|
|Xception|x|v|x|?|?|x|
|ResNeXt50|v|v|x|38,715,196|?|
|MobileNetV3|v|v|x|4,320,632|?|


## Requesties:

#### Keras:
- keras v2.2.4
#### Pytorch:
- pytorch v1.4
- [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning) [[doc]](https://pytorch-lightning.readthedocs.io/en/latest/)
- opencv


## Installation

This repo doesn't need any installation so far. Please clone the repo and directly import via LOCAL PATH.
```diff
$ git clone https://github.com/lsc-psd/modelib
```
[comment]: <> "Install via pip:"
[comment]: <> "```bash"
[comment]: <> "pip install efficientnet_pytorch"
[comment]: <> "```"
[comment]: <> "Or install from source:"
[comment]: <> "```bash"
[comment]: <> "git clone https://github.com/lukemelas/EfficientNet-PyTorch"
[comment]: <> "cd EfficientNet-Pytorch"
[comment]: <> "pip install -e ."
[comment]: <> "``` "


## Usage

[comment]: <> "#### Loading pretrained models"
### Example: Train from bash

```bash
python train.py -m VGG16 -f PATH_TO_TRAINDATA
"""
option: -m :        neural network structure
        -train :    path to train data folder, structure showed as below:
                    train_folder
                    ├─ Class1 (e.g. dog)
                    |   ├─ image1.jpg
                    |   └─ image2.jpg
                    └─ Class2 (e.g. cat)
                        ├─ image1.jpg
                        └─ image2.jpg
        -valid :    path to valid data folder, structure is same with train
"""        
```

### Example: Customize in python

Load a model in python:  
```python
from .models.SOME_MODEL import SOME_MODEL
```

## Contributing
If you want us to add other models, feel free to create an issue, or submit a pull request.
If you find a bug or having any question, create an issue and we'll solve it ASAP.

## LISCENCE
**WTFPL** licensed, just do what the fuck you want to public, cheers.

