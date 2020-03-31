import re
import os
import argparse
import importlib
from pytorch_lightning import Trainer

from .bin.model_builder import create_model
from .bin.utils import config_load


def main(args):
    config = config_load(args.config)

    # equals to from X import X
    structure = importlib.import_module(f'models.{config.m}')
    globals().update({'Structure': getattr(structure, structure.__dict__['__all__'])})

    # use def to create multiple inherited function
    system = create_model(structure, args.test, config.b)
    num_classes = len([x for x in os.listdir(config.f) if not re.match('^\.', x)])
    model = system(num_classes=num_classes)
    model = model.load_from_metrics(
        weights_path=args.ckpt,
        tags_csv=args.tags_csv,
        on_gpu=True,
        map_location=None
    )

    # most basic trainer, uses good defaults
    trainer = Trainer(gpus=1)
    trainer.test(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', default='./last_config.pickle', type=str, help='config file for models')
    parser.add_argument('-ckpt', type=str, help='checkpoint')
    parser.add_argument('-tags_csv', type=str, help='tags_csv file')
    parser.add_argument('-test', default='test_folder', type=str, help='data folder path')
    args = parser.parse_args()
    main(args)
