from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from model.InceptionV3 import InceptionV3

import os
import configparser

if __name__ == '__main__':
    config_ini = configparser.ConfigParser()
    config_ini.read('config.ini', encoding='utf-8')
    read_default = config_ini['DEFAULT']

    n_categories = int(read_default.get('n_categories'))
    train_dir = read_default.get('train_dir')
    checkpoint_path = read_default.get('checkpoint_path')
    img_height = int(read_default.get('img_height'))
    img_width = int(read_default.get('img_width'))
    input_shape = (img_height, img_width, 3)
    batch_size = int(read_default.get('batch_size'))
    nb_epochs = int(read_default.get('nb_epochs'))
    model_name = read_default.get('model_name')

    # model = exec(model_name+"()")
    # print(model)
    # print(type(model))
    model =InceptionV3(input_shape, int(n_categories))
    model.compile(optimizer=SGD(lr=0.005), loss='categorical_crossentropy', metrics=['accuracy'])
    # model.summary()

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode="nearest",
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training')

    val_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation')

    csv_logger = CSVLogger(checkpoint_path +'model.log')
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.3, patience=5, min_lr=1e-6)
    checkpointer = ModelCheckpoint(
        filepath=checkpoint_path + '/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
        save_best_only=True
    )
    early_stop = EarlyStopping(monitor='loss', patience=7, verbose=1, mode='auto')

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=train_generator.samples // batch_size,
                                  validation_data=val_generator,
                                  validation_steps=val_generator.samples // batch_size,
                                  epochs=nb_epochs,
                                  verbose=1,
                                  callbacks=[csv_logger, reduce_lr, checkpointer, early_stop]
                                  )