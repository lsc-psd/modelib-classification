from keras.layers import AveragePooling2D, GlobalAveragePooling2D, Concatenate
# can I do it?
class DenseNetSimple:
    def __init__(self, input_shape, nb_classes,
                 growth_rate=32, compression_factor=0.5, blocks=[6,12,24,18]):
        '''
        :param growth_rate: The number of filters to increase in DenseBlock
        :param compression_factor: The rate that compress in Transition layers.
        :param blocks: if we set (6,12,24,18), it's the same of DenseNet121.
        '''
        self.k = growth_rate
        self.compression = compression_factor
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.model = self.make_model(blocks)

    # DenseBlockのLayer
    def dense_block(self, input_tensor, input_channels, nb_blocks):
        x = input_tensor
        n_channels = input_channels
        for i in range(nb_blocks):
            # main line
            main = x
            # branch
            x = BatchNormalization()(x)
            x = Conv2D(128, (1, 1), activation='relu')(x)
            x = BatchNormalization()(x)
            x = Conv2D(self.k, (3, 3), padding="same", activation='relu')(x)
            # concatenate
            x = Concatenate()([main, x])
            n_channels += self.k
        return x, n_channels

    # Transition Layer
    def transition_layer(self, input_tensor, input_channels):
        n_channels = int(input_channels * self.compression)
        x = Conv2D(n_channels, (1, 1))(input_tensor)
        x = AveragePooling2D((2, 2))(x)
        return x, n_channels


    def make_model(self, blocks):
        inputs = Input(shape = self.input_shape)
        n = 16
        x = Conv2D(n, (1,1))(inputs)
        # DenseBlock - TransitionLayer - DenseBlock…
        for i in range(len(blocks)):
            # Transition
            if i != 0:
                x, n = self.transition_layer(x, n)
            # DenseBlock
            x, n = self.dense_block(x, n, blocks[i])
        x = GlobalAveragePooling2D()(x)
        x = Dropout(rate=0.3)(x)
        dense = Dense(512, activation = "relu")(x)
        output = Dense(self.nb_classes, activation = 'softmax', name='dense_3')(dense)
        # モデル
        densenet_model = Model(inputs=inputs, outputs=output)
        return densenet_model

# model = DenseNetSimple(input_shape=(128,128,3), nb_classes=3).model
