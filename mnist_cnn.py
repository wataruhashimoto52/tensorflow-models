# coding: utf-8

import numpy as np 
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot = False)

def conv_net(x_dict, n_classes, reuse, is_training):
    with tf.variable_scope("Convnet", reuse=reuse):
        x = x_dict['images']
        x = tf.reshape(x, shape=[-1, 28, 28, 1])

        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(conv1, 2, 2)
        conv2 = tf.layers.conv2d(x, 64, 3, activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(conv2, 2, 2)

        flat = tf.layers.flatten(pool2)

        fc1 = tf.layers.dense(flat, 1024)
        fc1 = tf.layers.dropout(fc1, rate=0.5, training=is_training)

        out = tf.layers.dense(fc1, n_classes)

    return out 

def model_fn(features, labels, mode):
    logits_train = conv_net(x_dict=features, n_classes=10, reuse=False, is_training=True)
    logits_test = conv_net(x_dict=features, n_classes=10, reuse=True, is_training=False)

    pred_classes = tf.argmax(logits_test, axis=1)
    prec_probs = tf.nn.softmax(logits_test)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    loss_op = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits_train, labels = tf.cast(labels, tf.int32)
        )
    )
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

    # acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss = loss_op,
        train_op = train_op,
        eval_metric_ops={'accuracy':acc_op}
    )

    return estim_specs


if __name__ == "__main__":
    model = tf.estimator.Estimator(model_fn)

    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'images':mnist.train.images},
        y = mnist.train.labels,
        batch_size=32,
        num_epochs=None,
        shuffle=True
    )

    model.train(input_fn, steps=100)

    # test 
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'images':mnist.test.images},
        y = mnist.test.labels,
        batch_size=32,
        shuffle=False
    )

    e = model.evaluate(input_fn)

    print("Test accuracy: {}".format(e['accuracy']))