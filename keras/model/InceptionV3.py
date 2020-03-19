import os
from keras.layers import AveragePooling2D, GlobalAveragePooling2D, Concatenate, Dense
from keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D
from keras.models import Model


def conv2d_bn(x, filters, kernel_size, strides=(1,1), padding='same'):
    x = Conv2D(filters,
               kernel_size,
               strides=strides,
               padding=padding)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def _InceptionV3_blockA(x):
    block1 = conv2d_bn(x, 64, (1,1))

    block2 = conv2d_bn(x, 48, (1,1))
    block2 = conv2d_bn(block2, 64, (5,5))

    block3 = conv2d_bn(x, 64, (1,1))
    block3 = conv2d_bn(block3, 96, (3,3))
    block3 = conv2d_bn(block3, 96, (3,3))

    block4 = AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same')(x)
    block4 = conv2d_bn(block4, 32, (1,1))

    x = Concatenate()([block1, block2, block3, block4])
    return x

# 2回繰り返す
def _InceptionV3_blockB(x):
    block1 = conv2d_bn(x, 64, (1, 1))

    block2 = conv2d_bn(x, 48, (1, 1))
    block2 = conv2d_bn(block2, 64, (5, 5))

    block3 = conv2d_bn(x, 64, (1, 1))
    block3 = conv2d_bn(block3, 96, (3, 3))
    block3 = conv2d_bn(block3, 96, (3, 3))

    block4 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    block4 = conv2d_bn(block4, 64, (1, 1))

    x = Concatenate()([block1, block2, block3, block4])
    return x

def _InceptionV3_blockC(x):
    block1 = conv2d_bn(x, 384, (3, 3), strides=(2,2), padding="valid")

    block2 = conv2d_bn(x, 64, (1, 1))
    block2 = conv2d_bn(block2, 96, (3, 3))
    block2 = conv2d_bn(block2, 96, (3,3), strides=(2,2), padding='valid')

    block3 = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = Concatenate()([block1, block2, block3])
    return x

def _InceptionV3_blockD(x):
    block1 = conv2d_bn(x, 192, (1, 1))

    block2 = conv2d_bn(x, 128, (1, 1))
    block2 = conv2d_bn(block2, 128, (1, 7))
    block2 = conv2d_bn(block2, 192, (7, 1))

    block3 = conv2d_bn(x, 128, (1, 1))
    block3 = conv2d_bn(block3, 128, (7, 1))
    block3 = conv2d_bn(block3, 128, (1, 7))
    block3 = conv2d_bn(block3, 128, (7, 1))
    block3 = conv2d_bn(block3, 192, (1, 7))

    block4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    block4 = conv2d_bn(block4, 192, (1, 1))

    x = Concatenate()([block1, block2, block3, block4])
    return x

def _InceptionV3_blockE(x):
    block1 = conv2d_bn(x, 192, (1, 1))

    block2 = conv2d_bn(x, 160, (1, 1))
    block2 = conv2d_bn(block2, 160, (1, 7))
    block2 = conv2d_bn(block2, 192, (7, 1))

    block3 = conv2d_bn(x, 160, (1, 1))
    block3 = conv2d_bn(block3, 160, (7, 1))
    block3 = conv2d_bn(block3, 160, (1, 7))
    block3 = conv2d_bn(block3, 160, (7, 1))
    block3 = conv2d_bn(block3, 192, (1, 7))

    block4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    block4 = conv2d_bn(block4, 192, (1, 1))

    x = Concatenate()([block1, block2, block3, block4])
    return x

def _InceptionV3_blockF(x):
    block1 = conv2d_bn(x, 192, (1, 1))

    block2 = conv2d_bn(x, 192, (1, 1))
    block2 = conv2d_bn(block2, 192, (1, 7))
    block2 = conv2d_bn(block2, 192, (7, 1))

    block3 = conv2d_bn(x, 192, (1, 1))
    block3 = conv2d_bn(block3, 192, (7, 1))
    block3 = conv2d_bn(block3, 192, (1, 7))
    block3 = conv2d_bn(block3, 192, (7, 1))
    block3 = conv2d_bn(block3, 192, (1, 7))

    block4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    block4 = conv2d_bn(block4, 192, (1, 1))

    x = Concatenate()([block1, block2, block3, block4])
    return x

def _InceptionV3_blockG(x):
    block1 = conv2d_bn(x, 192, (1, 1))
    block1 = conv2d_bn(block1, 320, (3,3), strides=(2,2), padding='valid')

    block2 = conv2d_bn(x, 192, (1, 1))
    block2 = conv2d_bn(block2, 192, (1, 7))
    block2 = conv2d_bn(block2, 192, (7, 1))
    block2 = conv2d_bn(block2, 192, (3, 3), strides=(2, 2), padding='valid')

    block3 = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = Concatenate()([block1, block2, block3])
    return x

def _InceptionV3_blockH(x):
    block1 = conv2d_bn(x, 320, (1, 1))

    block2 = conv2d_bn(x, 384, (1, 1))
    block2_1 = conv2d_bn(block2, 384, (1, 3))
    block2_2 = conv2d_bn(block2, 384, (3, 1))
    block2 = Concatenate()([block2_1, block2_2])

    block3 = conv2d_bn(x, 448, (1, 1))
    block3_1 = conv2d_bn(block3, 384, (3, 3))
    block3_2 = conv2d_bn(block3, 384, (1, 3))
    block3_3 = conv2d_bn(block3, 384, (3, 1))
    block3 = Concatenate()([block3_1, block3_2, block3_3])

    block4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    block4 = conv2d_bn(block4, 192, (1, 1))

    x = Concatenate()([block1, block2, block3, block4])
    return x


def build(input_shape, nb_classes):
    inputs = Input(shape=input_shape)
    x = conv2d_bn(inputs, 32, (3,3), (2,2), padding='valid')
    x = conv2d_bn(x, 32, (3,3), padding='valid')
    x = conv2d_bn(x, 64, (3,3))
    x = MaxPooling2D((3,3), strides=(2,2))(x)
    x = conv2d_bn(x, 80, (1,1), padding='valid')
    x = conv2d_bn(x, 192, (3,3), padding='valid')
    x = MaxPooling2D((3,3), strides=(2,2))(x)

    x = _InceptionV3_blockA(x)
    x = _InceptionV3_blockB(x)
    x = _InceptionV3_blockB(x) # two loop.
    x = _InceptionV3_blockC(x)
    x = _InceptionV3_blockD(x)
    x = _InceptionV3_blockE(x)
    x = _InceptionV3_blockE(x) # two loop.
    x = _InceptionV3_blockF(x)
    x = _InceptionV3_blockG(x)
    x = _InceptionV3_blockH(x)
    x = _InceptionV3_blockH(x) # two loop.

    x = GlobalAveragePooling2D()(x)
    x = Dense(nb_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    return model

# model = InceptionV3((299,299,3), 1000)
# model.summary()