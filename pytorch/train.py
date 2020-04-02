import re
import os
import argparse
import importlib
from configparser import ConfigParser
from pytorch_lightning import Trainer

from .bin.utils import param_save
from .bin.model_builder import create_model

os.chdir('/')


def main(config):
    # equals to from X import X
    structure = importlib.import_module('models.'+config['model_name'])
    globals().update({'Structure': getattr(structure, structure.__dict__['__all__'][0])})

    # use def to create multiple inherited function
    system = create_model(structure, config['train_dir'], config['val_dir'],
                          config['batch_size'], config['learning_rates'])
    num_classes = len([x for x in os.listdir(config['train_dir']) if not re.match('^\.', x)])
    model = system(num_classes=num_classes)
    # most basic trainer, uses good defaults
    trainer = Trainer(gpus=1,
                      default_save_path=config['ckpt_save_path'],
                      max_epochs=config['train_dir'])
    trainer.fit(model)
    param_save(config) # save config after finish training


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=str, help='config file')
    args = parser.parse_args()

    config = ConfigParser()
    config.read(args.c, encoding='utf-8')
    config = config['MODELIB']
    model = config['model_name']

    assert model in ['VGG16', 'DenseNet121', 'InceptionV3', 'MobileNetV3', 'ResNet50',
                      'ResNeXt50', 'VGG16', 'Xception'], \
        f'Chosen model {model} do not exist, please check model folder for available models.'

    main(config)
