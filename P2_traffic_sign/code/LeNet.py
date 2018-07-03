# updated LeNet to use the good parameter
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
import keras

Optimizer = "Adadelta" # Adadelta


class LeNet():
    # convlayer_filters, convlayer_kernel_size, padding, activation as input to make parameter tuning easier
    def __init__(self, num_classes, img_params):
        self.img_width, self.img_height, self.img_channels = img_params
        self.num_classes = num_classes

    def build(self):
        model = Sequential()
        # convolution, relu:  conv1 (?, 32, 32, 6)
        # max pool:  conv1 (?, 16, 16, 6)
        model.add(Conv2D(6, 5, activation='relu',padding='same', input_shape=(self.img_width, self.img_height, self.img_channels)))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        # convolution, relu:  conv2(?, 16, 16, 16)
        # max pool: conv2(?, 8, 8, 16)
        model.add(Conv2D(16, 5, activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # conv3 (?, 8, 8, 16)
        # conv3 (?, 4, 4, 16)
        model.add(Conv2D(16, 5, activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # Flatten
        # Flatten the output shape of the final pooling layer such that it's 1D instead of 3D.
        # fc0 (?, 256)
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation("relu"))
        # final softmax layer output prediction of probabilities for each class
        model.add(Dense(self.num_classes))
        model.add(Activation("softmax"))
        print("model summary")
        print(model.summary())
        if Optimizer == "Adam":
            model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.Adam(),
                          metrics=['accuracy'])
        elif (Optimizer == "Adadelta"):
            model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.Adadelta(),
                          metrics=['accuracy'])

        return model



