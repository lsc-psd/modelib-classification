from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping

import os

n_categories=len(os.listdir('../input'))
train_dir='../input'
file_name='./checkpoints'
img_height=224
img_width=224
input_shape=(img_height, img_width,3)
batch_size=64
nb_epochs=1

base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

x= base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
prediction = Dense(n_categories, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=prediction)

for layer in base_model.layers[:-4]:
    layer.trainable=False

model.compile(optimizer=SGD(lr=0.005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode="nearest",
    validation_split = 0.2
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size = batch_size,
    class_mode='categorical',
    subset = 'training')

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size = batch_size,
    class_mode='categorical',
    subset = 'validation')

csv_logger = CSVLogger('model.log')
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.3, patience=5, min_lr=1e-6)
checkpointer = ModelCheckpoint(
    filepath= file_name +'/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
    save_best_only=True
)
early_stop = EarlyStopping(monitor='loss', patience=7, verbose=1, mode='auto')


history = model.fit_generator(train_generator,
    steps_per_epoch= train_generator.samples// batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size,
    epochs = nb_epochs,
    verbose=1,
    callbacks=[csv_logger, reduce_lr, checkpointer, early_stop]
)