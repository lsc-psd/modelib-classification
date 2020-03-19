from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, Dropout
from keras.models import Model

class VGG16:
    def __init__(self, input_shape, nb_classes):
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.model = self.make_model()

    def make_model(self):
        inputs = Input(self.input_shape)
        x = Conv2D(64, (3,3), activation='relu', padding='same', name='block_conv1')(inputs)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block1_pool')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block2_pool')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block3_pool')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block4_pool')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block5_pool')(x)
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dropout(0.5, name='dropout1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dropout(0.5, name='dropout2')(x)
        outputs = Dense(self.nb_classes, activation='softmax', name='predictions')(x)
        VGGmodel = Model(inputs=inputs, outputs=outputs)
        return VGGmodel

def build(input_shape, nb_classes):
    model = VGG16(input_shape, nb_classes).model
    return model

# model = VGG16(input_shape=(128,128,3), nb_classes=10).model
