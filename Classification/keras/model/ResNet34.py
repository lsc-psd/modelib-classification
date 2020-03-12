from keras import backend as K
from keras.layers import Conv2D, Add, Input, BatchNormalization, Activation, MaxPooling2D, GlobalAveragePooling2D, Dense
from keras.models import Model

def shortcut(x, residual):
    x_shape = K.int_shape(x)
    residual_shape = K.int_shape(residual)
    if x_shape == residual_shape:
        shortcut = x
    else:
        shortcut = Conv2D(filters=residual_shape[3], kernel_size=(1, 1), strides=(2, 2))(x)
    return Add()([shortcut, residual])

def res_blocks(x,filter,stride):
    conv = Conv2D(filters=filter, kernel_size=(3, 3), strides=(stride, stride), padding="same", kernel_initializer='he_normal')(x)
    conv = BatchNormalization()(conv)
    conv = Activation("relu")(conv)
    conv = Conv2D(filters=filter, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal')(conv)
    conv = BatchNormalization()(conv)
    short_cut = shortcut(x, conv)
    conv = Activation("relu")(short_cut)
    return conv

def make_model(input_shape, num_classes):
    inputs = Input(input_shape)
    x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding="same", kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)
    x = res_blocks(x, 64, 1)
    x = res_blocks(x, 64, 1)
    x = res_blocks(x, 64, 1)
    x = res_blocks(x, 128, 2)
    x = res_blocks(x, 128, 1)
    x = res_blocks(x, 128, 1)
    x = res_blocks(x, 128, 1)
    x = res_blocks(x, 256, 2)
    x = res_blocks(x, 256, 1)
    x = res_blocks(x, 256, 1)
    x = res_blocks(x, 256, 1)
    x = res_blocks(x, 256, 1)
    x = res_blocks(x, 256, 1)
    x = res_blocks(x, 512, 2)
    x = res_blocks(x, 512, 1)
    x = res_blocks(x, 512, 1)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(units=num_classes, activation='softmax')(x)
    ResNetModel = Model(inputs=inputs, outputs=outputs)
    return ResNetModel