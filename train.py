from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, LeakyReLU
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import SGD, Adam
from keras.regularizers import l2
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import keras.backend as K
import glob
import os
from sklearn.utils import class_weight
from utils import MyFunction
from utils import Variables
import tensorflow as tf
import cv2

config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        per_process_gpu_memory_fraction=0.5,  # 最大値の50%まで
        allow_growth=True  # True->必要になったら確保, False->全部
    )
)
sess = tf.Session(config=config)


def train(args):
    img_width = args.width
    img_height = args.height
    batchSize = args.batchsize
    epoch = args.epoch
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    train_data_dir = os.path.join(ROOT_DIR, "DataSet_Periodontal_4567", "train")
    test_data_dir = os.path.join(ROOT_DIR, "DataSet_Periodontal_4567", "val")

    K.clear_session()

    # 訓練データ拡張
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )
    test_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
    )

    # 元になるモデルのダウンロード&最終層の作成
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    # base_model = VGG16(weights='imagenet', include_top=False)
    model = get_model(base_model)
    opt = SGD(lr=0.0001, momentum=0.9)
    # opt = Adam()
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])

    csv_logger = CSVLogger('model.log')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, min_lr=1e-6)
    checkpointer = ModelCheckpoint(
        filepath='./checkpoints/base_VGG16/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
        save_best_only=True
    )
    early_stop = EarlyStopping(monitor='val_loss', patience=7, verbose=1, mode='auto')

    x_train, y_train, train_label_count = load_data(train_data_dir, img_width, img_height)
    x_test, y_test, test_label_count = load_data(test_data_dir, img_width, img_height)
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)

    for i in range(5):
        kfold = StratifiedKFold(n_splits=5, shuffle=True)
        for count, (train, val) in enumerate(
                kfold.split(x_train[:train_label_count[0]+train_label_count[1]],
                            y_train[:train_label_count[0]+train_label_count[1]])):
            x_val_cv = np.concatenate([x_train[train], x_test[test_label_count[0]+test_label_count[1] + 1:], x_test[test_label_count[0]+test_label_count[1] + 1:]])
            y_val_cv = np.concatenate([y_train[train], y_test[test_label_count[0]+test_label_count[1] + 1:], y_test[test_label_count[0]+test_label_count[1] + 1:]])
            x_train_cv = np.concatenate([x_train[val], x_train[train_label_count[0]+train_label_count[1] + 1:], x_train[train_label_count[0]+train_label_count[1] + 1:]])
            y_train_cv = np.concatenate([y_train[val], y_train[train_label_count[0]+train_label_count[1] + 1:], y_train[train_label_count[0]+train_label_count[1] + 1:]])

            opt = SGD(lr=0.001 / (count + 1), momentum=0.9)
            model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])

            class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train_cv), y_train_cv)

            history = model.fit_generator(
                train_datagen.flow(x_train_cv, y_train_cv, batch_size=batchSize),
                steps_per_epoch=len(x_train_cv) // batchSize,
                class_weight=class_weights,
                epochs=30,
                shuffle=True,
                verbose=1,
                validation_data=test_datagen.flow(x_val_cv, y_val_cv),
                validation_steps=len(x_val_cv) // batchSize,
                callbacks=[reduce_lr, csv_logger, checkpointer])

    '''
    history = model.fit_generator(
        train_datagen.flow(x_train, y_train, batch_size=batchSize),
        steps_per_epoch=x_train.shape[0] / batchSize,
        class_weight=class_weights,
        epochs=epoch,
        validation_data=test_datagen.flow(x_test, y_test, batch_size=batchSize),
        validation_steps=x_test.shape[0] / batchSize,
        verbose=1,
        callbacks=[reduce_lr, csv_logger, checkpointer, early_stop]
    )
    '''
    MyFunction.print_Arruracy_progress(history)
    MyFunction.print_loss_progress(history)


def get_model(base_model):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    # x = LeakyReLU()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    # x = LeakyReLU()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, kernel_initializer="glorot_uniform", kernel_regularizer=l2(.001))(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # base_modelはweightsを更新する
    for layer in base_model.layers[:-8]:
        layer.trainable = False

    return model


def filename2label(filepath):
    temp = os.path.basename(os.path.dirname(filepath))
    value = int(temp)
    return value


def load_data(folderpath, img_width, img_height):
    x = []
    y = []
    labelpath = glob.glob(os.path.join(folderpath, "*"))
    label_count = []

    for label in labelpath:
        files = glob.glob(os.path.join(label, "*"))
        for n, file in enumerate(files):
            image = cv2.imread(file)
            h, w, c = image.shape
            # image = image[int(h / 3):h, 0:w]
            image = MyFunction.sharpen(image, img_width, img_height)
            # image = cv2.resize(image, (img_width, img_height))
            damage = filename2label(file)
            image = image / 255.0
            x.append(image)
            y.append(damage)
        label_count.append(n)
    x = np.array(x)
    y = np.array(y)
    return x, y, label_count


if __name__ == '__main__':
    args = Variables.Args().get_args().parse_args()
    train(args)
