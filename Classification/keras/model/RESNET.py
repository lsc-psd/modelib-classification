from keras import backend as K
from keras.layers import Conv2D,Add,Input,BatchNormalization,Activation,MaxPooling2D,GlobalAveragePooling2D,Dense
from keras.models import Model
from keras.regularizers import l2

def resnet_first_root(x):
    first_root = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding="same", kernel_initializer="he_normal")(x)
    first_root = BatchNormalization()(first_root)
    first_root = Activation('relu')(first_root)
    first_pool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(first_root)
    return first_pool

def resnet_final_root(x,classes):
    last_root = BatchNormalization()(x)
    last_root = Activation('relu')(last_root)
    last_pool = GlobalAveragePooling2D()(last_root)
    outputs = Dense(units=classes, kernel_initializer='he_normal', activation='sigmoid')(last_pool)
    return outputs

def base_block1_1(x,filters):
    root = BatchNormalization()(x)
    root = Activation("relu")(root)
    root = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding="same",
                  kernel_initializer="he_normal", kernel_regularizer=l2(1.e-4))(root)
    return root

def base_block3_3(x,filters):
    root = BatchNormalization()(x)
    root = Activation('relu')(root)
    root = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding="same",
                  kernel_initializer="he_normal", kernel_regularizer=l2(1.e-4))(root)
    return root

def shortcut(x, residual):
    x_shape = K.int_shape(x)
    residual_shape = K.int_shape(residual)
    #if x_shape == residual_shape:
        #shortcut = x
    #else:
    stride_w = int(round(x_shape[1] / residual_shape[1]))
    stride_h = int(round(x_shape[2] / residual_shape[2]))
    shortcut = Conv2D(filters=residual_shape[3], kernel_size=(1, 1), strides=(stride_w, stride_h),
                      kernel_initializer='he_normal')(x)
    return Add()([shortcut, residual])

def building_block(x,filters,first_layer):
    if first_layer == "first_layer":
        conv1 = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding="same",
                       kernel_initializer="he_normal", kernel_regularizer=l2(1.e-4))(x)
    else:
        conv1 = base_block3_3(x, filters)
    conv2 = base_block3_3(conv1, filters)
    return shortcut(x, conv2)

def bottleneck_block(x,filters,first_layer):
    if first_layer == "first_layer":
        conv1 = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding="same",
                       kernel_initializer="he_normal", kernel_regularizer=l2(1.e-4))(x)
    else:
        conv1 = base_block1_1(x, filters)
    conv2 = base_block3_3(conv1, filters)
    conv3 = base_block1_1(conv2, filters*4)
    return shortcut(x, conv3)

def make_base_model(input_shape, num_class, repetitions):
    inputs = Input(input_shape)
    throughput = resnet_first_root(inputs)
    for i in range(repetitions[0]):
        if i == 0:
            throughput = building_block(throughput, 64, "first_layer")
        else:
            throughput = building_block(throughput, 64, "None")
    for i in range(repetitions[1]):
        throughput = building_block(throughput, 128, "None")
    for i in range(repetitions[2]):
        throughput = building_block(throughput, 256, "None")
    for i in range(repetitions[3]):
        throughput = building_block(throughput, 512, "None")
    outputs = resnet_final_root(throughput, num_class)
    ResNetModel = Model(inputs=inputs, outputs=outputs)
    return ResNetModel

def make_bottleneck_model(input_shape, num_class, repetitions):
    inputs = Input(input_shape)
    throughput = resnet_first_root(inputs)
    for i in range(repetitions[0]):
        if i == 0:
            throughput = bottleneck_block(throughput, 64, "first_layer")
        else:
            throughput = bottleneck_block(throughput, 64, "None")
    for i in range(repetitions[1]):
        throughput = bottleneck_block(throughput, 128, "None")
    for i in range(repetitions[2]):
        throughput = bottleneck_block(throughput, 256, "None")
    for i in range(repetitions[3]):
        throughput = bottleneck_block(throughput, 512, "None")
    outputs = resnet_final_root(throughput, num_class)
    ResNetModel = Model(inputs=inputs, outputs=outputs)
    return ResNetModel

def resnet_18(input_shape, num_class):
    return make_base_model(input_shape, num_class, [2, 2, 2, 2])

def resnet_34(input_shape, num_class):
    return make_base_model(input_shape, num_class, [3, 4, 6, 3])

def resnet_50(input_shape, num_class):
    return make_bottleneck_model(input_shape, num_class, [3, 4, 6, 3])

def resnet_101(input_shape, num_class):
    return make_bottleneck_model(input_shape, num_class, [3, 4, 23, 3])

def resnet_152(input_shape, num_class):
    return make_bottleneck_model(input_shape, num_class, [3, 8, 36, 3])
=======
from keras import backend as K
from keras.layers import Conv2D,Add,Input,BatchNormalization,Activation,MaxPooling2D,GlobalAveragePooling2D,Dense
from keras.models import Model
from keras.regularizers import l2

def resnet_first_root(x):
    first_root = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding="same", kernel_initializer="he_normal")(x)
    first_root = BatchNormalization()(first_root)
    first_root = Activation('relu')(first_root)
    first_pool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(first_root)
    return first_pool

def resnet_final_root(x,classes):
    last_root = BatchNormalization()(x)
    last_root = Activation('relu')(last_root)
    last_pool = GlobalAveragePooling2D()(last_root)
    outputs = Dense(units=classes, kernel_initializer='he_normal', activation='sigmoid')(last_pool)
    return outputs

def base_block1_1(x,filters):
    root = BatchNormalization()(x)
    root = Activation("relu")(root)
    root = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding="same",
                  kernel_initializer="he_normal", kernel_regularizer=l2(1.e-4))(root)
    return root

def base_block3_3(x,filters):
    root = BatchNormalization()(x)
    root = Activation('relu')(root)
    root = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding="same",
                  kernel_initializer="he_normal", kernel_regularizer=l2(1.e-4))(root)
    return root

def shortcut(x, residual):
    x_shape = K.int_shape(x)
    residual_shape = K.int_shape(residual)
    #if x_shape == residual_shape:
        #shortcut = x
    #else:
    stride_w = int(round(x_shape[1] / residual_shape[1]))
    stride_h = int(round(x_shape[2] / residual_shape[2]))
    shortcut = Conv2D(filters=residual_shape[3], kernel_size=(1, 1), strides=(stride_w, stride_h),
                      kernel_initializer='he_normal')(x)
    return Add()([shortcut, residual])

def building_block(x,filters,first_layer):
    if first_layer == "first_layer":
        conv1 = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding="same",
                       kernel_initializer="he_normal", kernel_regularizer=l2(1.e-4))(x)
    else:
        conv1 = base_block3_3(x, filters)
    conv2 = base_block3_3(conv1, filters)
    return shortcut(x, conv2)

def bottleneck_block(x,filters,first_layer):
    if first_layer == "first_layer":
        conv1 = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding="same",
                       kernel_initializer="he_normal", kernel_regularizer=l2(1.e-4))(x)
    else:
        conv1 = base_block1_1(x, filters)
    conv2 = base_block3_3(conv1, filters)
    conv3 = base_block1_1(conv2, filters*4)
    return shortcut(x, conv3)

def make_base_model(input_shape, num_class, repetitions):
    inputs = Input(input_shape)
    throughput = resnet_first_root(inputs)
    for i in range(repetitions[0]):
        if i == 0:
            throughput = building_block(throughput, 64, "first_layer")
        else:
            throughput = building_block(throughput, 64, "None")
    for i in range(repetitions[1]):
        throughput = building_block(throughput, 128, "None")
    for i in range(repetitions[2]):
        throughput = building_block(throughput, 256, "None")
    for i in range(repetitions[3]):
        throughput = building_block(throughput, 512, "None")
    outputs = resnet_final_root(throughput, num_class)
    ResNetModel = Model(inputs=inputs, outputs=outputs)
    return ResNetModel

def make_bottleneck_model(input_shape, num_class, repetitions):
    inputs = Input(input_shape)
    throughput = resnet_first_root(inputs)
    for i in range(repetitions[0]):
        if i == 0:
            throughput = bottleneck_block(throughput, 64, "first_layer")
        else:
            throughput = bottleneck_block(throughput, 64, "None")
    for i in range(repetitions[1]):
        throughput = bottleneck_block(throughput, 128, "None")
    for i in range(repetitions[2]):
        throughput = bottleneck_block(throughput, 256, "None")
    for i in range(repetitions[3]):
        throughput = bottleneck_block(throughput, 512, "None")
    outputs = resnet_final_root(throughput, num_class)
    ResNetModel = Model(inputs=inputs, outputs=outputs)
    return ResNetModel

def resnet_18(input_shape, num_class):
    return make_base_model(input_shape, num_class, [2, 2, 2, 2])

def resnet_34(input_shape, num_class):
    return make_base_model(input_shape, num_class, [3, 4, 6, 3])

def resnet_50(input_shape, num_class):
    return make_bottleneck_model(input_shape, num_class, [3, 4, 6, 3])

def resnet_101(input_shape, num_class):
    return make_bottleneck_model(input_shape, num_class, [3, 4, 23, 3])

def resnet_152(input_shape, num_class):
    return make_bottleneck_model(input_shape, num_class, [3, 8, 36, 3])
