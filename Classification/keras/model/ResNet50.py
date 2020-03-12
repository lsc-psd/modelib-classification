from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Input, BatchNormalization

class VGG16:
    def __init__(self, input_shape, nb_classes):
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.model = self.make_model(blocks)

        def identity_block(self):
            return 3

        def make_model(self):
            inputs = Input(self.input_shape)
            x = ZeroPadding2D(padding=(3,3), name='conv1_pad')(inputs)
            x = Conv2D(64, (7,7), strides=(2,2), padding='valid', kernel_initializer='he_normal', name='conv1')(x)
            x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)



# model = DenseNetSimple(input_shape=(128,128,3), nb_classes=3).model