# updated LeNet to use the good parameter
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import LeakyReLU
from keras.layers import Dense, Dropout, Activation
import keras

class LeNet():
    def __init__(self, num_classes, img_params):
        self.img_width, self.img_height, self.img_channels = img_params
        self.num_classes = num_classes

    def build(self):
        model = Sequential()
        model.add(Conv2D(6, 5, activation='relu',padding='valid', input_shape=(self.img_width, self.img_height, self.img_channels)))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        model.add(Conv2D(16, 5, activation='relu', padding='valid'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(128))
        model.add(LeakyReLU())
        model.add(Dropout(0.5))
        model.add(Dense(84, activation='sigmoid'))
        model.add(Dense(self.num_classes))
        model.add(Activation("softmax"))
        print("model summary")
        print(model.summary())
        model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.Adam(),
                          metrics=['accuracy'])
        return model
