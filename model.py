from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D 
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import ELU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
import numpy as np 
import cv2


class VGGNet:
    @staticmethod
    def build():
        model = Sequential()
        model.add(Conv2D(32, (3,3), padding = "same",kernel_initializer = "he_normal", input_shape = (48,48,1)))
        model.add(ELU())
        model.add(BatchNormalization(axis= -1))

        model.add(Conv2D(32, (3,3), padding = "same",kernel_initializer = "he_normal"))
        model.add(ELU())
        model.add(BatchNormalization(axis= -1))

        model.add(MaxPool2D(pool_size = (2,2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3,3), kernel_initializer = "he_normal", padding = "same"))
        model.add(ELU())
        model.add(BatchNormalization(axis = -1))

        model.add(Conv2D(64, (3,3), kernel_initializer = "he_normal", padding = "same"))
        model.add(ELU())
        model.add(BatchNormalization(axis = -1))

        model.add(MaxPool2D(pool_size = (2,2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3,3), kernel_initializer = "he_normal", padding = "same"))
        model.add(ELU())
        model.add(BatchNormalization(axis=-1))

        model.add(Conv2D(128, (3,3), kernel_initializer = "he_normal", padding = "same"))
        model.add(ELU())
        model.add(BatchNormalization(axis = -1))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(64, kernel_initializer = "he_normal"))
        model.add(ELU())
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(64, kernel_initializer = "he_normal"))
        model.add(ELU())
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(7, kernel_initializer = "he_normal"))
        model.add(Activation("softmax"))

        return model 


class Preprocess:
    @staticmethod
    def preprocess(image):
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        gray = np.array(gray,dtype='float32')
        gray /= 255
        gray_resized = cv2.resize(gray, (48,48), interpolation = cv2.INTER_AREA)
        final = np.reshape(gray_resized, (1,48, 48,1))

        return final

