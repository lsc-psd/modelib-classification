from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, Dropout
from keras.models import Model


def vgg_blocks(x, filters, repetitions):
    for i in range(repetitions):
        x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)
    return x

def vgg_final_root(x, nb_classes):
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(nb_classes, activation='softmax')(x)
    return x


def make_model16(input_shape, nb_classes):
    inputs = Input(input_shape)
    x = vgg_blocks(inputs, 64, 2)
    x = vgg_blocks(x, 128, 2)
    x = vgg_blocks(x, 256, 3)
    x = vgg_blocks(x, 512, 3)
    x = vgg_blocks(x, 512, 3)
    outputs = vgg_final_root(x, nb_classes)
    VGGmodel = Model(inputs=inputs, outputs=outputs)
    return VGGmodel

def make_model19(input_shape, nb_classes):
    inputs = Input(input_shape)
    x = vgg_blocks(inputs, 64, 2)
    x = vgg_blocks(x, 128, 2)
    x = vgg_blocks(x, 256, 4)
    x = vgg_blocks(x, 512, 4)
    x = vgg_blocks(x, 512, 4)
    outputs = vgg_final_root(x, nb_classes)
    VGGmodel = Model(inputs=inputs, outputs=outputs)
    return VGGmodel