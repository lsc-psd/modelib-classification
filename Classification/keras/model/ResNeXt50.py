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

def res_blocks(x, filter, stride):
    conv = Conv2D(filters=filter, kernel_size=(1, 1), strides=(stride, stride), padding="same", kernel_initializer='he_normal')(x)
    conv = BatchNormalization()(conv)
    conv = Activation("relu")(conv)
    conv = Conv2D(filters=filter, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal')(conv)
    conv = BatchNormalization()(conv)
    conv = Activation("relu")(conv)
    conv = Conv2D(filters=filter*4, kernel_size=(1, 1), strides=(1,1), padding="same", kernel_initializer='he_normal')(conv)
    conv = BatchNormalization()(conv)
    return conv


class ResNeXt50:
    def __init__(self, input_shape, nb_classes):
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.model = self.make_model(blocks)

    def make_model(self):
        inputs = Input(self.input_shape)
        x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding="same", kernel_initializer='he_normal')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)

        for j in range(3):
            x1_array = [0 for k in range(32)]
            for i in range(32):
                x1_array[i] = res_blocks(x, 64, 2)
                x_all = Add()(x1_array)
                all = Add()([x, x_all])
                x = Activation("relu")(all)

        for j in range(4):
            x2_array = [0 for k in range(32)]
            for i in range(32):
                x2_array[i] = res_blocks(x, 64, 2)
                x_all = Add()(x2_array)
                all = Add()([x, x_all])
                x = Activation("relu")(all)

