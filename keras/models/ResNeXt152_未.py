from keras.layers import Conv2D, MaxPooling2D, Lambda, Input, Dense, Flatten

class ResNeXt152:
    def __init__(self, input_shape, nb_classes):
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.model = self.make_model()

    def make_model(self):
        inputs = Input(self.input_shape)

        outputs = Dense(self.nb_classes, activation='softmax', name='predictions')(x)
        ResNeXtmodel = Model(inputs=inputs, outputs=outputs)
        return ResNeXtmodel

def build(input_shape, nb_classes):
    return ResNeXt152(input_shape, nb_classes).model

# model = ResNeXt152(input_shape=(128,128,3), nb_classes=1000).model