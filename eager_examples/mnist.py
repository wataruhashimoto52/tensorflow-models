# coding: utf-8

import numpy as np 
import tensorflow as tf
import dataset
import tensorflow.contrib.eager as tfe 

class MNISTModel(tf.keras.Model):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(units=10)
        self.dense2 = tf.keras.layers.Dense(units=10)
        self.dense3 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        """
        Run the model.
        """ 
        result = self.dense1(inputs)
        result = self.dense2(result)
        result = self.dense3(result)
        
        return result 

tf.enable_eager_execution()
model = MNISTModel()

batch = tf.zeros([1,1,784])

result = model(batch)
print(result)

dataset_train = dataset.train('./datasets').shuffle(60000).repeat(4).batch(32)

def loss(model, x, y):
    prediction = model(x)
    return tf.losses.sparse_softmax_cross_entropy(
            labels=y, logits=prediction
        )

def grad(model, inputs, targets):
    with tfe.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return tape.gradient(loss_value, model.variables)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

x, y = tfe.Iterator(dataset_train).next()

print("Initial loss: {:.3f}".format(loss(model, x, y)))

for (i, (x,y)) in enumerate(tfe.Iterator(dataset_train)):
    # calculate derivatives of the inputs function with respect to its parameters.
    grads = grad(model, x, y)

    # apply the gradient to the model. 
    optimizer.apply_gradients(zip(grads, model.variables),
    global_step=tf.train.get_or_create_global_step())

    if i % 200 == 0:
        print("Loss at step {:04d}: {:.3f}".format(i, loss(model, x, y)))

print("Final loss: {:.3f}".format(loss(model, x, y)))