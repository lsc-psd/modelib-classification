import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import mnist
from keras.optimizers import sgd
from keras.callbacks import Callback, CSVLogger
import argparse
#import tensorflow as tf
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#tf.keras.backend.set_session(tf.Session(config=config))

class PlotLosses(Callback):

    def on_train_begin(self, logs={}):
        self.epoch_cnt = 0
        plt.axis([0, self.epochs, 0, 0.25])
        plt.ion()

    def on_train_end(self, logs={}):
        plt.ioff()
        plt.legend(['loss', 'val_loss'], loc='best')
        plt.show()

    def on_epoch_end(self, epoch, logs={}):
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        x = self.epoch_cnt
        plt.scatter(x, loss, c='b', label='loss')
        plt.scatter(x, val_loss, c='r', label='val_loss')
        plt.pause(0.05)
        self.epoch_cnt += 1

def plot_loss_result(history):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    learning_count = len(loss) + 1

    plt.plot(range(1, learning_count), loss, marker="+", label="loss")
    plt.plot(range(1, learning_count), val_loss, marker=".", label="val_loss")
    plt.legend(loc="best", fontsize=10)
    plt.xlabel("learning_count")
    plt.ylabel("loss")
    plt.savefig('./keras_cnn_loss.jpg')
    plt.show()

def plot_acc_result(history):

    acc = history.history["acc"]
    val_acc = history.history["val_acc"]

    learning_count = len(acc) + 1

    plt.plot(range(1, learning_count), acc, marker="+", label="acc")
    plt.plot(range(1, learning_count), val_acc, marker=".", label="val_acc")
    plt.legend(loc="best", fontsize=10)
    plt.xlabel("learning_count")
    plt.ylabel("acc")
    plt.savefig('./keras_cnn_acc.jpg')
    plt.show()

def main(epochs=50,batch_size=128,learning_rate=0.01):

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)/255
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)/255
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    print(x_train.shape)
    print(x_test.shape)

    model = Sequential()
    model.add(Conv2D(10, kernel_size=(5, 5), activation="relu", input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(20, (5, 5), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=100, activation="relu"))
    model.add(Dense(units=10, activation="softmax"))
    model.summary()

    model.compile(loss="mean_squared_error",
                  optimizer=sgd(lr=learning_rate),
                  metrics=["accuracy"])

    plot_losses = PlotLosses()
    plot_losses.epochs = epochs
    csv_logger = CSVLogger('./keras_cnn_trainlog.csv')

    history = model.fit(x_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=1,
                        validation_data=(x_test, y_test),
                        callbacks=[plot_losses, csv_logger])

    score = model.evaluate(x_test, y_test, verbose=1)
    print("test loss:", score[0])
    print("test accuracy:", score[1])

    plot_loss_result(history)
    plot_acc_result(history)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='MNIST')
    parser.add_argument('--epochs', '--e', dest='epochs', type=int, help='size of epochs', default=50)
    parser.add_argument('--batch_size', '--b', dest='batch_size', type=int, help='size of batch', default=128)
    parser.add_argument('--learning_rate', '--l', dest='learning_rate', type=float, help='size of learning_rate', default=0.01)
    args = parser.parse_args()

    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    main(epochs, batch_size, learning_rate)