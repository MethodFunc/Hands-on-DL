from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Reshape
from keras.datasets import fashion_mnist
# PCA

# encoder = Sequential([Dense(2, input_shape=[3])])
# decoder = Sequential([Dense(3, input_shape=[2])])
# autoencoder = Sequential([encoder, decoder])
#
# autoencoder.compile(keras.optimizers.SGD(lr=0.1), "mse")
#
# # history = autoencoder.fit(X_train, X_train, epochs=20)
# # codings = encoder.predict(X_train)

## Stacked autoencoder

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train /= 255.0
x_test /= 255.0


stacked_encoder = Sequential([
    Flatten(input_shape=[28, 28]),
    Dense(100, activation='selu'),
    Dense(30, activation='selu')
])

stacked_decoder = Sequential([
    Dense(100, activation='selu', input_shape=[30]),
    Dense(28 * 28, activation='sigmoid'),
    Reshape([28, 28])
])

stacked_ae = Sequential([stacked_encoder, stacked_decoder])
stacked_ae.compile(keras.optimizers.SGD(lr=1.5), keras.losses.binary_crossentropy)

history = stacked_ae.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))