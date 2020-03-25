import os
import argparse
import importlib
from pytorch_lightning import Trainer

from .bin.utils import config_save
from .bin.model_builder import create_model

os.chdir('/')


def main(args):
    # equals to from X import X
    structure = importlib.import_module(f'models.{args.m}')
    globals().update({'Structure': getattr(structure, structure.__dict__['__all__'][0])})

    # use def to create multiple inherited function
    system = create_model(structure, args.train, args.valid, args.b)
    model = system()
    # most basic trainer, uses good defaults
    trainer = Trainer(gpus=1,
                      default_save_path=f'{args.save}',
                      max_epochs=args.max_epoch)
    trainer.fit(model)
    config_save(args) # save config after finish training


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', default='VGG16', type=str, help='which models')
    parser.add_argument('-train', default='train_imgs', type=str, help='train data folder path')
    parser.add_argument('-valid', default='train_imgs', type=str, help='valid data folder path')
    parser.add_argument('-b', default=32, type=int, help='batch size')
    parser.add_argument('-save', default='ckpt', type=str, help='folder to save checkpoint')
    parser.add_argument('-max_epoch', default=500, type=int, help='max epoch count')
    args = parser.parse_args()

    assert args.m in ['VGG16', 'DenseNet121', 'InceptionV3', 'MobileNetV3', 'ResNet50',
                      'ResNeXt50', 'VGG16', 'Xception'], \
        f'Chosen model {args.m} do not exist, please check model folder for available models.'

    main(args)
