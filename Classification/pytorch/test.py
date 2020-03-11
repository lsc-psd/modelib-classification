import argparse
import importlib
from pytorch_lightning import Trainer

from .models.model_builder import create_model
from .utils import config_load


def main(args):
    config = config_load(args.config)

    # equals to from X import X
    Structure = importlib.import_module(f'models.{config.m}')
    globals().update({'Structure': getattr(Structure, Structure.__dict__['__all__'])})

    # use def to create multiple inherited function
    System = create_model(Structure, config.f, config.b)
    model = System()
    model = model.load_from_metrics(
        weights_path=args.ckpt,
        tags_csv=args.ckpt,
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
    parser.add_argument('-f', default='test_imgs', type=str, help='data folder path')
    args = parser.parse_args()
    main(args)