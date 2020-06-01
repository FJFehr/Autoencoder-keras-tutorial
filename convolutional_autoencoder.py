# Autoencoder tutorial
# Fabio Fehr
# 1 June 2020

# This code shows the construction of a non-linear AE being used for the MNIST dataset
# Since our inputs are images, it makes sense to use convolutional neural networks (convnets) as encoders and decoders.
# these are just better on images. MUCH SLOWER must have taken about 20 minutes to do 50 epochs

# The encoder will consist in a stack of Conv2D and MaxPooling2D layers
# (max pooling being used for spatial down-sampling)

# the decoder will consist in a stack of Conv2D and UpSampling2D layers.

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
import matplotlib.pyplot as plt
from keras import backend as K

input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


###########################################################################################################
# Data preparation ########################################################################################
###########################################################################################################


from keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()

# MNIST digits with shape (samples, 3, 28, 28), and we will just normalize pixel values between 0 and 1.
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format


###########################################################################################################
# Train AE ################################################################################################
###########################################################################################################

# First, let's open up a terminal and start a TensorBoard server that will read logs stored at /tmp/autoencoder.
# tensorboard --logdir=/tmp/autoencoder
#  In the callbacks list we pass an instance of the TensorBoard callback. After every epoch,
#  this callback will write logs to /tmp/autoencoder, which can be read by our TensorBoard server.

from keras.callbacks import TensorBoard

autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

# This allows us to monitor training in the TensorBoard web interface (by navighating to http://0.0.0.0:6006):


###########################################################################################################
# Visualisation of output #################################################################################
###########################################################################################################

decoded_imgs = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# We can also have a look at the 128-dimensional encoded representations.
# These representations are 8x4x4, so we reshape them to 4x32 in order to be able to display them as grayscale images.
