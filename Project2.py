import tensorflow as tf
import keras
from keras import layers

model = keras.Sequential()

model.add(layers.Input(shape = (7,10)))
model.add(layers.Flatten())
model.add(layers.Dense(3,activation="sigmoid", name = "Layer1"))
model.add(layers.Dense(10, activation = "relu", name = "Layer2" ))


x = tf.ones((4,7,10))
y = model(x)

wnb = model.layers[0].get_weights()

model.summary()
print(wnb[0].shape)