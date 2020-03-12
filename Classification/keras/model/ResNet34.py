from keras import backend as K
from keras.layers import Conv2D,Add,Input,BatchNormalization,Activation,MaxPooling2D,GlobalAveragePooling2D,Dense
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
    conv = Conv2D(filters=filter, kernel_size=(3, 3), strides=(stride,stride), padding="same", kernel_initializer='he_normal')(x)
    conv = BatchNormalization()(conv)
    conv = Activation("relu")(conv)
    conv = Conv2D(filters=filter, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal')(conv)
    conv = BatchNormalization()(conv)
    short_cut = shortcut(x, conv)
    conv = Activation("relu")(short_cut)
    return conv

def make_model(input_shape, num_classes):
    inputs = Input(input_shape)
    conv1 = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding="same", kernel_initializer='he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    first_pool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)

    conv2 = res_blocks(first_pool, 64, 1)
    conv2 = res_blocks(conv2, 64, 1)
    conv2 = res_blocks(conv2, 64, 1)

    conv3 = res_blocks(conv2, 128, 2)
    conv3 = res_blocks(conv3, 128, 1)
    conv3 = res_blocks(conv3, 128, 1)
    conv3 = res_blocks(conv3, 128, 1)

    conv4 = res_blocks(conv3, 256, 2)
    conv4 = res_blocks(conv4, 256, 1)
    conv4 = res_blocks(conv4, 256, 1)
    conv4 = res_blocks(conv4, 256, 1)
    conv4 = res_blocks(conv4, 256, 1)
    conv4 = res_blocks(conv4, 256, 1)

    conv5 = res_blocks(conv4, 512, 2)
    conv5 = res_blocks(conv5, 512, 1)
    conv5 = res_blocks(conv5, 512, 1)

    last_pool = GlobalAveragePooling2D()(conv5)
    outputs = Dense(units=num_classes, activation='softmax')(last_pool)
    ResNetModel = Model(inputs=inputs, outputs=outputs)
    return ResNetModel
