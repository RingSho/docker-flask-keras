# from __future__ import print_function

import time
import keras

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

from PIL import Image
import numpy as np
import cv2

class MnistModel:
    def __init__(self):
        self.num_classes = 10
        self.model = self.model_mnist(mode = "test")

    def model_mnist(self, mode="train"):
        model = Sequential()
        model.add(Dense(512, activation='relu', input_shape=(784,)))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.num_classes, activation='softmax'))
        if mode == "train":
            model.compile(loss='categorical_crossentropy',
                        optimizer=RMSprop(),
                        metrics=['accuracy'])
        print("Done load model.")
        return model

    def train(self, save_dir=""):
        batch_size = 64
        epochs = 1

        # the data, split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train.reshape(60000, 784)
        x_test = x_test.reshape(10000, 784)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        self.model = self.model_mnist(mode = "train")

        history = self.model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_data=(x_test, y_test))
        score = self.model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        self.model.save(save_dir + 'model.h5')
        return {"loss": str(score[0]), "acc": str(score[1])}

    def predict(self, weight_path = "mnist/model.h5", image_path="images/input/00000.png"):
        self.model.load_weights(weight_path)
        print("Done loaded weights")
        # 画像の読み込みとreshape
        X_test = []
        target_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        target_img = cv2.resize(target_img, (28, 28))
        X_test.append(target_img)
        X_test = np.array(X_test)
        X_test = X_test.reshape(1, 784)

        result = self.model.predict(X_test)
        predict_number = result.argmax()
        return {"predict": str(predict_number)}

if __name__ == "__main__":
    mnist_model = MnistModel()
    # mnist_model.train()
    print(mnist_model.predict("mnist/model.h5", "images/input/00000.png"))