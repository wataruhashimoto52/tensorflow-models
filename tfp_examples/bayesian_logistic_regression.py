# coding: utf-8

import numpy as np 
from absl import flags 
import tensorflow as tf 
import tensorflow_probability as tfp 
import matplotlib.pyplot as plt 
import seaborn as sns 
from matplotlib import cm 
from matplotlib import figure 
from matplotlib.backends import backend_agg 

tfd = tf.contrib.distributions

flags.DEFINE_float("learning_rate",
                   default=0.01,
                   help="Initial learning rate.")
flags.DEFINE_integer("max_steps",
                     default=1500,
                     help="Number of training steps to run.")
flags.DEFINE_integer("batch_size",
                     default=32,
                     help="Batch size. Must divide evenly into dataset sizes.")
flags.DEFINE_string(
    "model_dir",
    default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"),
                         "logistic_regression/"),
    help="Directory to put the model's fit.")
flags.DEFINE_integer("num_examples",
                     default=256,
                     help="Number of datapoints to generate.")
flags.DEFINE_integer("num_monte_carlo",
                     default=50,
                     help="Monte Carlo samples to visualize weight posterior.")

FLAGS = flags.FLAGS

def toy_logistic_data(num_examples, input_size=2, weights_prior_stddev=5.0):
    """Generates synthetic data for binary classification.
    Args:
        num_examples: The number of samples to generate (scalar Python `int`).
        input_size: The input space dimension (scalar Python `int`).
        weights_prior_stddev: The prior standard deviation of the weight
        vector. (scalar Python `float`).
    Returns:
        random_weights: Sampled weights as a Numpy `array` of shape
        `[input_size]`.
        random_bias: Sampled bias as a scalar Python `float`.
        design_matrix: Points sampled uniformly from the cube `[-1,
        1]^{input_size}`, as a Numpy `array` of shape `(num_examples,
        input_size)`.
        labels: Labels sampled from the logistic model `p(label=1) =
        logistic(dot(inputs, random_weights) + random_bias)`, as a Numpy
        `int32` `array` of shape `(num_examples, 1)`.
    """
    random_weights = weights_prior_stddev * np.random.randn(input_size)
    random_bias = np.random.randn()
    design_matrix = np.random.rand(num_examples, input_size) * 2 - 1
    logits = np.reshape(
        np.dot(design_matrix, random_weights) + random_bias,
        (-1, 1))
    p_labels = 1. / (1 + np.exp(-logits))
    labels = np.int32(p_labels > np.random.rand(num_examples, 1))
    return random_weights, random_bias, np.float32(design_matrix), labels

def build_input_pipeline(x, y, batch_size):
    """Build a Dataset iterator for supervised classification.
    Args:
        x: Numpy `array` of inputs, indexed by the first dimension.
        y: Numpy `array` of labels, with the same first dimension as `x`.
        batch_size: Number of elements in each training batch.
    Returns:
        batch_data: `Tensor` feed  inputs, of shape
        `[batch_size] + x.shape[1:]`.
        batch_labels: `Tensor` feed of labels, of shape
        `[batch_size] + y.shape[1:]`.
    """

    training_dataset = tf.data.Dataset.from_tensor_slices((x, y))
    training_batches = training_dataset.repeat().batch(batch_size)
    training_iterator = training_batches.make_one_shot_iterator()
    batch_data, batch_labels = training_iterator.get_next()
    return batch_data, batch_labels


def main(argv):
    del argv 

    if tf.gfile.Exists(FLAGS.log_dir):
        tf.logging.warn(
            "Warning: deleting old log directory at {}".format(
                FLAGS.log_dir))
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)

    w_true, b_true, x, y = toy_logistic_data(FLAGS.num_examples, 2)

    with tf.Graph().as_default():
        inputs, labels = build_input_pipeline(x, y, FLAGS.batch_size)

        with tf.name_scope("logistic_regression", values=[inputs]):

            layer = tfp.layers.DenseFlipout(
                units=1,
                activation=None,
                kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(),
                bias_posterior_fn=tfp.layers.default_mean_field_normal_fn()
            )
            logits = layer(input)
            labels_dist = tfd.Bernoulli(logits=logits)

            # compute the -elbo as the loss, averaged over the batch size 
            neg_log_likelihood = -tf.reduce_mean(labels_dist.log_prob(labels))
            kl = sum(layer.losses) / FLAGS.num_examples
            elbo_loss = neg_log_likelihood + kl 