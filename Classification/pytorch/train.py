import os
import re
import argparse
import importlib
import cv2

from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from torch import optim
import torch

import pytorch_lightning as pl
from pytorch_lightning import Trainer

os.chdir('./')


class ClassificationDataset(Dataset):
    def __init__(self, folder_path, imgsize=(256, 256), annotation_file=None):

        self.folder = folder_path
        self.imgsize = imgsize
        self.d_list = dict(filenames=[], labels=[], labels_str=[])
        self._list_prep()
        # ##TrainVal split
        # if annotation_file is not None:
        #     with open(annotation_file) as f:
        #         content = f.readlines()
        #     self.files = [x.strip() for x in content]
        # else:
        #     self.files = None

    def __getitem__(self, index):
        img, label = self.pull_item(index)
        return img, label

    def __len__(self):
        return len(self.d_list['labels'])

    def _list_prep(self):
        classes = [x for x in os.listdir(self.folder) if not re.match('^\.',x)]
        for i, _class in enumerate(classes):
            filenames = os.listdir(os.path.join(self.folder, _class))
            self.d_list['filenames'] += [os.path.join(self.folder, _class, x) for x in filenames]
            self.d_list['labels'] += [i] * len(filenames)
            self.d_list['labels_str'] += [_class] * len(filenames)

    def pull_item(self, index):
        img_file = self.d_list['filenames'][index]
        label = self.d_list['labels'][index]
        img = cv2.imread(img_file, 1)/255
        img = img.astype('float32')
        img = cv2.resize(img, self.imgsize)

        return torch.from_numpy(img).permute(2, 0, 1), torch.tensor(label)


def create_model(Structure, folder_path):
    class TrainModel(Structure, pl.LightningModule):
        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self.forward(x)
            loss = F.cross_entropy(y_hat, y)
            tensorboard_logs = {'train_loss': loss}
            return {'loss': loss, 'log': tensorboard_logs}

        def configure_optimizers(self):
            # can return multiple optimizers and learning_rate schedulers
            return optim.Adam(self.parameters(), lr=0.02)

        def train_dataloader(self):
            # REQUIRED
            return DataLoader(ClassificationDataset(folder_path),
                              batch_size=32)
    return TrainModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", default='VGG16', help='which models')
    parser.add_argument("-f", default='train_imgs', help='data folder path')
    args = parser.parse_args()

    assert args.m in ['VGG16', 'DenseNet121', 'InceptionV3', 'MobileNetV3', 'ResNet50',
                      'ResNeXt50', 'VGG16', 'Xception'], \
        f'Chosen model {args.m} do not exist, please check model folder for available models.'

    # equals to from X import X
    Structure = importlib.import_module(f'models.{args.m}')
    globals().update({'Structure': getattr(Structure, Structure.__dict__['__all__'])})

    System = create_model(Structure, args.f)
    model = System()

    # most basic trainer, uses good defaults
    trainer = Trainer(gpus=1)
    trainer.fit(model)


