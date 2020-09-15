from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping

import os
import glob
import configparser
import argparse
from importlib import import_module

def generate(train_dir, val_dir, img_height, img_width, batch_size):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode="nearest"
        # validation_split=0.2
    )
    val_datagen = ImageDataGenerator(
        rescale=1.0/255
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical'
        # subset='training'
        )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical'
        # subset='validation'
        )

    return train_generator, val_generator

def callbacks(checkpoint_path):
    csv_logger = CSVLogger(checkpoint_path +'model.log')
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.3, patience=5, min_lr=1e-6)
    checkpointer = ModelCheckpoint(
        filepath=checkpoint_path + '/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
        save_best_only=True)
    early_stop = EarlyStopping(monitor='loss', patience=7, verbose=1, mode='auto')
    return csv_logger, reduce_lr, checkpointer, early_stop


def train(read_default):
    # 各種変数の定義
    print(read_default.get('train_dir'))
    n_categories = len(glob.glob(os.path.join(read_default.get('train_dir'), "*")))
    train_dir = read_default.get('train_dir')
    val_dir = read_default.get('val_dir')
    checkpoint_path = read_default.get('checkpoint_path')
    img_height = int(read_default.get('img_height'))
    img_width = int(read_default.get('img_width'))
    input_shape = (img_height, img_width, 3)
    batch_size = int(read_default.get('batch_size'))
    nb_epochs = int(read_default.get('nb_epochs'))
    learning_rates = float(read_default.get('learning_rates'))
    model_name = read_default.get('model_name')

    # 画像増幅
    train_generator, val_generator =generate(
        train_dir, val_dir, img_height, img_width, batch_size)

    # モデルの読み込みとコンパイル
    Structure = import_module(f'models.{model_name}')
    model = Structure.build(input_shape, n_categories)
    model.compile(optimizer=SGD(lr=learning_rates), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()


    # コールバック関数の設定
    csv_logger, reduce_lr, checkpointer, early_stop = callbacks(checkpoint_path)

    # 学習
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=train_generator.samples // batch_size,
                                  validation_data=val_generator,
                                  validation_steps=val_generator.samples // batch_size,
                                  epochs=nb_epochs,
                                  verbose=1,
                                  callbacks=[csv_logger, reduce_lr, checkpointer, early_stop]
                                  )
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', default='config.ini', type=str, help='config file')
    args = parser.parse_args()
    config_ini = configparser.ConfigParser()
    config_ini.read(args.c, encoding='utf-8')
    read_default = config_ini['MODELIB']

    train(read_default)