import keras.datasets
import keras.datasets.mnist
import keras.losses
import keras.metrics
import keras.optimizers
import keras.utils
import tensorflow as tf
import keras
from keras import layers

model = keras.Sequential(
    [
        layers.Input(shape= (28,28,1)),
        layers.Conv2D(filters= 3, kernel_size= (5,5), padding= "same", activation= "relu", name = "ConvLayer_1"),
        layers.MaxPool2D(pool_size= (2,2)),
        layers.Conv2D(filters = 9, kernel_size= (3,3), activation= "relu", padding= "same", name = "ConvLayer_2"),
        layers.MaxPool2D(pool_size= (2,2)),
        layers.Flatten(),
        layers.Dense(100, activation= "relu", name = "FCLayer_1"),
        layers.Dense(10, name= "OutputLayer")
    ]
)



model.compile(
    loss= keras.losses.SparseCategoricalCrossentropy(from_logits= True),
    optimizer= keras.optimizers.SGD(),
    metrics= [keras.metrics.SparseCategoricalAccuracy()]
)


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape((-1,28,28,1)).astype("float32") / 255
x_test = x_test.reshape((-1,28,28,1)).astype("float32") / 255
#y_train = keras.utils.to_categorical(y_train, num_classes= 10)
#y_test = keras.utils.to_categorical(y_test, num_classes= 10)

history = model.fit(x_train, y_train, batch_size= 32, epochs = 2, validation_split= 0.15, shuffle= True, verbose= 1)

_ ,test_scores = model.evaluate(x_test, y_test, verbose = 2)

print("Test Accuracy: ", test_scores)