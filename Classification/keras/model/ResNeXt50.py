from keras import backend as K
from keras.layers import Conv2D, Add, Input, BatchNormalization, Activation, MaxPooling2D, GlobalAveragePooling2D, Dense
from keras.models import Model

def shortcut(x, residual):
    x_shape = K.int_shape(x)
    residual_shape = K.int_shape(residual)
    if x_shape == residual_shape:
        shortcut = x
    else:
        stride_w = int(round(x_shape[1] / residual_shape[1]))
        stride_h = int(round(x_shape[2] / residual_shape[2]))
        shortcut = Conv2D(filters=residual_shape[3], kernel_size=(1, 1), strides=(stride_w, stride_h))(x)
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
                x1_array[i] = res_blocks(x, 64, 1)
            x1_all = Add()(x1_array)
            shortcut1 = shortcut(x, x1_all)
            x = Activation("relu")(shortcut1)

        for j in range(4):
            x2_array = [0 for k in range(32)]
            if j == 0:
                st = 2
            else:
                st = 1
            for i in range(32):
                x2_array[i] = res_blocks(x, 128, st)
            x2_all = Add()(x2_array)
            shortcut2 = shortcut(x, x2_all)
            x = Activation("relu")(shortcut2)

        for j in range(6):
            x3_array = [0 for k in range(32)]
            if j == 0:
                st = 2
            else:
                st = 1
            for i in range(32):
                x3_array[i] = res_blocks(x, 256, st)
            x3_all = Add()(x3_array)
            shortcut3 = shortcut(x, x3_all)
            x = Activation("relu")(shortcut3)

        for j in range(3):
            x4_array = [0 for k in range(32)]
            if j == 0:
                st = 2
            else:
                st = 1
            for i in range(32):
                x4_array[i] = res_blocks(x, 512, st)
            x4_all = Add()(x4_array)
            shortcut4 = shortcut(x, x4_all)
            x = Activation("relu")(shortcut4)

        x = GlobalAveragePooling2D()(x)
        outputs = Dense(units=self.nb_classes, activation='softmax')(x)
        ResNeXtModel = Model(inputs=inputs, outputs=outputs)
        return ResNeXtModel
