# Autoencoder tutorial
# Fabio Fehr
# 1 June 2020

# This code shows the construction of a non-linear AE being used for the MNIST dataset
# This has an added regularisation term for sparcity we notice in training the loss
# starts off much higher and does not go down as low. only 50 epochs its still poor
# 100 epochs and with he regularisation its less likely to overfit. Even then its poor


# experiments regulariser = 10e-5, epochs = 200 (still poor loss = 0.27)
# experiments regulariser = 10e-1, epochs = 50 (even worse loss = 0.29)
# experiments regulariser = 10e-10, epochs = 50 (much better loss = 0.104)


from keras.layers import Input, Dense
from keras.models import Model

from keras import regularizers

encoding_dim = 32

input_img = Input(shape=(784,))
# add a Dense layer with a L1 activity regularizer
encoded = Dense(encoding_dim, activation='relu',
                activity_regularizer=regularizers.l1(10e-10))(input_img)
decoded = Dense(784, activation='sigmoid')(encoded)

autoencoder = Model(input_img, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))

# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]

# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

# per-pixel binary crossentropy loss, and the Adam optimizer:
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

###########################################################################################################
# Data preparation ########################################################################################
###########################################################################################################

from keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()

# normalize all values between 0 and 1 and we will flatten the 28x28 images into vectors of size 784
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

###########################################################################################################
# Train AE ################################################################################################
###########################################################################################################

autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

###########################################################################################################
# Visualisation of output #################################################################################
###########################################################################################################

import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()