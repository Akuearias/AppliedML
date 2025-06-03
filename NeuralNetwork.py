import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import load_sample_image
import numpy as np

fashion_mnist = keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# A simple deep NN for classification, using Keras
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28))) # No params, O = 28 * 28 = 784
model.add(keras.layers.Dense(300, activation='relu')) # Params = (28 * 28 + 1) * 300 = 235500, O = 300
model.add(keras.layers.Dense(100, activation='relu')) # Params = (300 + 1) * 100 = 30100, O = 100
model.add(keras.layers.Dense(10, activation='softmax')) # Params = (100 + 1) * 10 = 1010, O = 10

# Total params = 235500 + 30100 + 1010 = 266610, all parameters are trainable.


# A simple deep NN for regression, using Keras
housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

model = keras.models.Sequential([
    keras.layers.Dense(30, activation='relu', input_shape=X_train.shape[1:]),
    keras.layers.Dense(1)
])

model.compile(loss='mean_squared_error', optimizer='sgd')
history = model.fit(X_train, y_train, epochs=20,
                    validation_data=(X_val, y_val))
MSE_test = model.evaluate(X_test, y_test)
X_new = X_test[:1]
y_pred = model.predict(X_new)
print(y_pred)


# Functional API version of NN model
input_ = keras.layers.Input(shape=X_train.shape[1:]) # O = 8, params = 0
hidden_1 = keras.layers.Dense(30, activation='relu')(input_) # O = 30, params = 30 * (8 + 1) = 270
hidden_2 = keras.layers.Dense(30, activation='relu')(hidden_1) # O = 30, params = 30 * (30 + 1) = 930
concat = keras.layers.Concatenate()([input_, hidden_2]) # O = 30 + 8 = 38, params = 0
output = keras.layers.Dense(1)(concat) # O = 1, params = 38 + 1 = 39
model = keras.Model(inputs=[input_], outputs=[output])

model.compile(loss=['mse', 'mse'], loss_weights=[0.9, 0.1], optimizer='sgd')


# Batch normalization
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), # O = 28*28=784, params = 0
    keras.layers.BatchNormalization(), # O = 784, params = 784 * 4 = 3136
    keras.layers.Dense(300, activation='elu', kernel_initializer='he_normal'), # O = 300, params = (784 + 1) * 300 = 235500
    keras.layers.BatchNormalization(), # O = 300, params = 300 * 4 = 1200
    keras.layers.Dense(100, activation='elu', kernel_initializer='he_normal'), # O = 100, params = (300 + 1) * 100 = 30100
    keras.layers.BatchNormalization(), # O = 100, params = 100 * 4 = 400
    keras.layers.Dense(10, activation='softmax') # O = 10, params = (100 + 1) * 10 = 1010
])
# Total params = 3136 + 235500 + 1200 + 30100 + 400 + 1010 = 271346, with (3136 + 1200 + 400) // 2 = 2368 non-trainable params.

# Reusing pretrained layers (Only a sample code illustrating how to do so)
'''
model_A = keras.models.load_model("my_model_A.h5")
model_B_on_A = keras.models.Sequential(model_A.layers[:-1])
model_B_on_A.add(keras.layers.Dense(1, activation="sigmoid"))

for layer in model_B_on_A.layers[:-1]:
    layer.trainable = False

model_B_on_A.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])
history = model_B_on_A.fit(X_train, y_train, epochs=4, validation_data=(X_val, y_val))

for layer in model_B_on_A.layers[:-1]:
    layer.trainable = True

optimizer = keras.optimizers.SGD(lr=1e-4)# the default lr is 1e-2
model_B_on_A.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=["accuracy"])
history = model_B_on_A.fit(X_train, y_train, epochs=16, validation_data=(X_val,y_val))
'''
# CNN - Tensorflow
china = load_sample_image("china.jpg")
flower = load_sample_image("flower.jpg")
images = np.array([china, flower])
images = tf.cast(images, tf.float32)
batch_size, height, width, channels = images.shape

filters = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
filters[:, 3, :, 0] = 1
filters[:, 3, :, 1] = 1

Os = tf.nn.conv2d(images, filters, strides=1, padding="SAME")


# CNN - Keras Conv2D Layers

np.random.seed(42)
tf.random.set_seed(42)

conv = keras.layers.Conv2D(filters=2, kernel_size=7, strides=1, padding='SAME', activation="relu", input_shape = images.shape[1:])
conv_O = conv(images)
print(conv_O.shape)

# CNN architecture example
model = keras.models.Sequential([
    keras.layers.Conv2D(64,7,activation="relu", padding="same", input_shape=[28, 28, 1]), # O = 28 * 28 * 64, params = (7*7+1)*64=3200
    keras.layers.MaxPooling2D(2), # O = 28/2 * 28/2 * 64, params = 0
    keras.layers.Conv2D(128,3, activation="relu", padding="same"), # O = 14 * 14 * 128, params = (3*3*64+1) * 128 = 73856
    keras.layers.Conv2D(128,3, activation="relu", padding="same"), # O = 14 * 14 * 128, params = (3*3*128+1) * 128 = 147584
    keras.layers.MaxPooling2D(2), # O = 7 * 7 * 128, params = 0
    keras.layers.Conv2D(256,3, activation="relu", padding="same"), # O = 7 * 7 * 256, params = (3 * 3 * 128 + 1) * 256 = 295168
    keras.layers.Conv2D(256,3, activation="relu", padding="same"), # O = 7 * 7 * 256, params = (3 * 3 * 256 + 1) * 256 = 590080
    keras.layers.MaxPooling2D(2), # O = 3 * 3 * 256, params = 0
    keras.layers.Flatten(), # O = 2304, params = 0
    keras.layers.Dense(128, activation="relu"), # O = 128, params = (2304 + 1) * 128 = 295040
    keras.layers.Dropout(0.5), # O = 128, params = 0
    keras.layers.Dense(64, activation="relu"), # O = 64, params (128 + 1) * 64 = 8256
    keras.layers.Dropout(0.5), # O = 64, params = 0
    keras.layers.Dense(10, activation="softmax") # O = 10, params = (64 + 1) * 10 = 650
])
# Total params is the sum of parameters of all layers, i.e. 1413834.
# There are no non-trainable parameters.