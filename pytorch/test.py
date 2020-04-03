import re
import os
import argparse
import importlib
from configparser import ConfigParser
from pytorch_lightning import Trainer

from .bin.model_builder import create_model
from .bin.utils import param_load


def main(args, config):
    param = param_load(args.train_param)

    # equals to from X import X
    structure = importlib.import_module('models.'+param['model_name'])
    globals().update({'Structure': getattr(structure, structure.__dict__['__all__'])})

    # use def to create multiple inherited function
    system = create_model(structure, config['test_dir'], param['batch_size'])
    num_classes = len([x for x in os.listdir(param['train_dir']) if not re.match('^\.', x)])
    model = system(num_classes=num_classes)
    model = model.load_from_metrics(
        weights_path=config['checkpoint_path'],
        tags_csv=config['tags_csv'],
        on_gpu=True,
        map_location=None
    )

    # most basic trainer, uses good defaults
    trainer = Trainer(gpus=1)
    trainer.test(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_param', default='./last_config.pickle', type=str, help='config file for models')
    parser.add_argument('-c', type=str, help='config file')
    args = parser.parse_args()

    config = ConfigParser()
    config.read(args.c, encoding='utf-8')
    config = config['MODELIB']

    main(args, config)
