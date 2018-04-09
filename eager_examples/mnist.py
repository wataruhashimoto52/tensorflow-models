# coding: utf-8

import numpy as np 
import keras
import tensorflow as tf
import dataset
import tensorflow.contrib.eager as tfe 
from keras.datasets import mnist 

num_classes = 10
class MNISTModel(tf.keras.Model):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(units=512, input_shape=(784,))
        self.dense2 = tf.keras.layers.Dense(units=512)
        self.dense3 = tf.keras.layers.Dense(units=num_classes)

    def call(self, inputs):
        """
        Run the model.
        """ 
        result = self.dense1(inputs)
        result = self.dense2(result)
        result = self.dense3(result)
        
        return tf.nn.softmax(result)

tf.enable_eager_execution()
model = MNISTModel()

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
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

model.compile(optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.001),
            loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=32,verbose=1,
            validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print("score:", score)