from keras import backend as K
from keras.layers import Conv2D, Add, Input, BatchNormalization, Activation, MaxPooling2D, GlobalAveragePooling2D,SeparableConv2D, Dense
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

def block(x,in_filter,filter,out_filter,stride=1,cardinality=32):
    res_x = Conv2D(filters=filter, kernel_size=(1, 1), strides=(stride, stride), padding="same", kernel_initializer='he_normal')(x)
    res_x = BatchNormalization()(res_x)
    res_x = Activation("relu")(res_x)
    multiplier = filter // cardinality
    res_x = SeparableConv2D(filters=filter, kernel_size=(3, 3), strides=(1, 1), padding="same",depth_multiplier=multiplier,  kernel_initializer='he_normal')(res_x)
    res_x = BatchNormalization()(res_x)
    res_x = Activation("relu")(res_x)
    res_x = Conv2D(filters=out_filter, kernel_size=(1, 1), strides=(1,1), padding="same", kernel_initializer='he_normal')(res_x)
    res_x = BatchNormalization()(res_x)
    if in_filter != out_filter:
        x = Conv2D(out_filter, [1, 1], strides=1, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
    if stride == 2:
        x = MaxPooling2D([2, 2], strides=2, padding="same")(x)
    x = Add()([res_x,x])
    x = Activation("relu")(x)
    return x


class ResNeXt50:
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

        x = block(x, 64, 64, 256)
        x = block(x, 256, 64, 256)
        x = block(x, 256, 64, 256)

        x = block(x, 256, 128, 512, stride=2)
        x = block(x, 512, 128, 512)
        x = block(x, 512, 128, 512)
        x = block(x, 512, 128, 512)

        x = block(x, 512, 256, 1024, stride=2)
        x = block(x, 1024, 256, 1024)
        x = block(x, 1024, 256, 1024)
        x = block(x, 1024, 256, 1024)
        x = block(x, 1024, 256, 1024)
        x = block(x, 1024, 256, 1024)

        x = block(x, 1024, 512, 2018, stride=2)
        x = block(x, 1024, 512, 2018)
        x = block(x, 1024, 512, 2018)

        x = GlobalAveragePooling2D()(x)
        outputs = Dense(units=self.nb_classes, activation='softmax')(x)
        ResNeXtModel = Model(inputs=inputs, outputs=outputs)
        return ResNeXtModel

def build(input_shape, nb_classes):
    return ResNeXt50(input_shape, nb_classes).model

# model = ResNeXt50(input_shape=(224,224,3), nb_classes=1000).model
