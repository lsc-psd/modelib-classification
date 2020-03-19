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
    conv = Conv2D(filters=filter*4, kernel_size=(1, 1), strides=(1, 1), padding="same", kernel_initializer='he_normal')(conv)
    conv = BatchNormalization()(conv)
    short_cut = shortcut(x, conv)
    conv = Activation("relu")(short_cut)
    return conv

class ResNet101:
    def __init__(self, input_shape, nb_classes):
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.model = self.make_model()

    def make_model(self):
        inputs = Input(self.input_shape)
        x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding="same", kernel_initializer='he_normal')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)
        for i in range(3):
            x = res_blocks(x, 64, 1)
        for i in range(4):
            if i == 0:
                x = res_blocks(x, 128, 2)
            else:
                x = res_blocks(x, 128, 1)
        for i in range(23):
            if i == 0:
                x = res_blocks(x, 256, 2)
            else:
                x = res_blocks(x, 256, 1)
        for i in range(3):
            if i == 0:
                x = res_blocks(x, 512, 2)
            else:
                x = res_blocks(x, 512, 1)
        x = GlobalAveragePooling2D()(x)
        outputs = Dense(units=self.nb_classes, activation='softmax')(x)
        ResNetModel = Model(inputs=inputs, outputs=outputs)
        return ResNetModel

def build(input_shape, nb_classes):
    return ResNet101(input_shape, nb_classes).model

# model = ResNet101(input_shape=(224,224,3), nb_classes=1000).model