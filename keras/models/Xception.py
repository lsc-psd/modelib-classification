from keras.layers import Conv2D, MaxPooling2D, Input, Add, Dense, BatchNormalization, Activation, SeparableConv2D, GlobalAveragePooling2D
from keras.models import Model

def shortcut(x, filters):
    x = Conv2D(filters, (1, 1), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    return x

def xception_blockA(x, filters):
    x = Conv2D(filters, (3, 3), strides=2)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def xception_blockB(x, filters, maxpooling=False):
    x = SeparableConv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    if maxpooling:
        x = MaxPooling2D((3, 3), strides=2, padding='same')(x)
    else:
        x = Activation('relu')(x)
    return x

def xception_blockC(x, filters, maxpooling=False):
    x = Activation('relu')(x)
    x = SeparableConv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    if maxpooling:
        x = MaxPooling2D((3, 3), strides=2, padding='same')(x)
    return x


class Xception:
    def __init__(self, input_shape, nb_classes):
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.model = self.make_model()

    def make_model(self):
        inputs = Input(self.input_shape)
        x = xception_blockA(inputs, 32)
        x = xception_blockA(x, 64)
        residual = shortcut(x, 128)
        x = xception_blockB(x, 128)
        x = xception_blockB(x, 128, maxpooling=True)
        x = Add()([x, residual])
        residual = shortcut(x, 256)
        x = xception_blockC(x, 256)
        x = xception_blockC(x, 256, maxpooling=True)
        x = Add()([x, residual])
        residual = shortcut(x, 728)
        x = xception_blockC(x, 728)
        x = xception_blockC(x, 728, maxpooling=True)
        x = Add()([x, residual])

        for i in range(8):
            residual = x
            x = xception_blockC(x, 728)
            x = xception_blockC(x, 728)
            x = xception_blockC(x, 728)
            x = Add()([x, residual])

        residual = shortcut(x, 1024)
        x = xception_blockC(x, 728)
        x = xception_blockC(x, 1024, maxpooling=True)
        x = Add()([x, residual])
        x = xception_blockB(x, 1536)
        x = xception_blockB(x, 2048)
        x = GlobalAveragePooling2D()(x)
        outputs = Dense(units=self.nb_classes, activation='softmax')(x)
        XceptionModel = Model(inputs=inputs, outputs=outputs)
        return XceptionModel

def build(input_shape, nb_classes):
    return Xception(input_shape, nb_classes).model

# model = Xception(input_shape=(224,224,3), nb_classes=1000).model