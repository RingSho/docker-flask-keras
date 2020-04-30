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

def model_mnist(mode="train"):
    num_classes = 10
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    if mode == "train":
        model.compile(loss='categorical_crossentropy',
                    optimizer=RMSprop(),
                    metrics=['accuracy'])
    return model

def mnist2_train(save_dir=""):
    batch_size = 1
    num_classes = 10
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
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = model_mnist()

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    # score = [0.1, 99.9]
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    time.sleep(1)
    model.save(save_dir + 'model.h5')
    return {"loss": score[0], "acc": score[1]}

def mnist2_predict(weight_path="mnist2/model.h5", image_path="5_test.png"):
    model = model_mnist(mode = "test")
    model.load_weights(weight_path)

    # 画像の読み込みとreshape
    X_test = []
    target_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    target_img = cv2.resize(target_img, (28, 28))
    X_test.append(target_img)
    X_test = np.array(X_test)
    X_test = X_test.reshape(1, 784)

    result = model.predict(X_test)
    predict_number = result.argmax()
    return {"predict": str(predict_number)}

if __name__ == "__main__":
    # mnist2_train()
    print(mnist2_predict("mnist2/model.h5", "5_test.png"))
    print(mnist2_predict("mnist2/model.h5", "5_test.png"))