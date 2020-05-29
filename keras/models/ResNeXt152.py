from keras.layers import Conv2D, MaxPooling2D, Input, Add, Dense, BatchNormalization, Activation, SeparableConv2D, GlobalAveragePooling2D
from keras.models import Model
from keras import backend as K

def shortcut(conv, residual):
    conv_shape = K.int_shape(conv)
    residual_shape = K.int_shape(residual)
    if conv_shape != residual_shape:
        residual = Conv2D(filters=conv_shape[3], kernel_size=(1, 1), strides=(2, 2))(residual)
    return Add()([conv, residual])

def block(x,filter,stride=1,cardinality=32):
    multiplier = filter // cardinality
    conv = Conv2D(filters=filter, kernel_size=(1, 1), strides=(stride, stride), padding="same", kernel_initializer='he_normal')(x)
    conv = BatchNormalization()(conv)
    conv = Activation("relu")(conv)
    conv = SeparableConv2D(filters=filter, kernel_size=(3, 3), strides=(1, 1), padding="same", depth_multiplier=multiplier, kernel_initializer='he_normal')(conv)
    conv = BatchNormalization()(conv)
    conv = Activation("relu")(conv)
    conv = Conv2D(filters=filter*4, kernel_size=(1, 1), strides=(1, 1), padding="same", kernel_initializer='he_normal')(conv)
    conv = BatchNormalization()(conv)
    conv = shortcut(conv, x)
    conv = Activation("relu")(conv)
    return conv


class ResNeXt152:
    def __init__(self, input_shape, nb_classes):
        self.input_shape = input_shape
        self.nb_classes = nb_classes

    def make_model(self):
        inputs = Input(self.input_shape)
        x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding="same",
                   kernel_initializer='he_normal')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)

        x = block(x, 64)
        x = block(x, 64)
        x = block(x, 64)

        x = block(x, 128, stride=2)
        x = block(x, 128)
        x = block(x, 128)
        x = block(x, 128)
        x = block(x, 128)
        x = block(x, 128)
        x = block(x, 128)

        x = block(x, 256, stride=2)
        for i in range(35):
            x = block(x, 256)

        x = block(x, 512, stride=2)
        x = block(x, 512)
        x = block(x, 512)

        x = GlobalAveragePooling2D()(x)
        outputs = Dense(units=self.nb_classes, activation='softmax')(x)
        ResNeXtModel = Model(inputs=inputs, outputs=outputs)
        return ResNeXtModel

def build(input_shape, nb_classes):
    return ResNeXt152(input_shape, nb_classes).model

# model = ResNeXt152(input_shape=(224,224,3), nb_classes=1000).model