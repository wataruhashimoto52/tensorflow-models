

import numpy as np 
import tensorflow as tf 
import argparse, os, sys 
import tensorflow_probability as tfp 
from weight_uncertainty import *
tfd = tf.contrib.distributions 

# plot
import matplotlib 
matplotlib.use("Agg")
from matplotlib import figure 
from matplotlib.backend import backend_agg 
import seaborn as sns 

# data
from tensorflow.contrib.learn.python.learn.datasets import mnist 

IMAGE_SHAPE = (28, 28)

def build_input_pipeline(mnist_data, batch_size, heldout_size):
    training_dataset = tf.data.Dataset.from_tensor_slices(
        (mnist_data.train.images, np.int32(mnist_data.train.labels)))
    training_batches = training_dataset.repeat().batch(batch_size)
    training_iterator = training_batches.make_one_shot_iterator()

    # build iterator over the heldout set with batch_size=heldout_size,
    # i.e., return the entire heldout set as a constant
    heldout_dataset = tf.data.Dataset.from_tensor_slices(
        (mnist_data.validation.images,
        np.int32(mnist_data.validation.labels)))
    heldout_frozen = (heldout_dataset.take(heldout_size).
                      repeat().batch(heldout_size))
    heldout_iterator = heldout_frozen.make_one_shot_iterator()

    handle = tf.placeholder(tf.string, shape=[])
    feedable_iterator = tf.data.Iterator.from_string_handle(
        handle, training_batches.output_types, training_batches.output_shapes)
    images, labels = feedable_iterator.get_next()

    return images, labels, handle, training_iterator, heldout_iterator

def build_fake_data():
    pass
    

def run_training():
    
    if FLAGS.fake_data:
        mnist_data = build_fake_data()
    else:
        mnist_data = mnist.read_data_sets(FLAGS.input_data_dir)

    

    def build_bayesian_nn_model(inputs):
        with tf.name_scope(name, "build_bayesian_nn", [inputs]):
            net = inputs 
            for layer_size in FLAGS.encoder_layers:
                net = tf.layers.dense(net, layer_size, activation=FLAGS.activation)
            logits = tf.layers.dense(net, 10, activation=None) # 10クラス分類
            model = tfd.Categorical(logits=logits)

            return model

    with tf.Graph().as_default():
        (images, labels, handle,
        training_iterator, heldout_iterator) = build_input_pipeline(
            mnist_data.FLAGS.batch_size, mnist_data.validation.num_examples)

    

def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.logging.warn(
            "Warning: deleting old log directory at {}".format(
                FLAGS.log_dir))
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)

    run_training()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Initial learning rate."
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1500,
        help="Number of training steps to run."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size.  Must divide evenly into the dataset sizes."
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=os.path.join(os.getenv("TEST_TMPDIR", "images"),
                            "bayesian_nn/"),
        help="Directory to put the log data."
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=256,
        help="Number of datapoints to generate."
    )
    parser.add_argument(
        "--viz_steps",
        type=int,
        default=400,
        help="Frequency at which save visualizations."
    )
    parser.add_argument(
        "--encoder_layers",
        type=str,
        default="128,32",
        help="comma-separated list of layer sizes for the encoder."
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        help="Activation function for the encoder and decoder networks."
    )
    parser.add_argument(
        "--n_monte_carlo",
        type=int,
        default=25,
        help="Monte Carlo samples used to visualize the weight posterior"
    )
    parser.add_argument(
        "--fake_data",
        default=False,
        action="store_true",
        help="If true, uses fake data for unit testing."
    )
    parser.add_argument(
        "--input_data_dir",
        type=str,
        default=os.path.join(os.getenv("TEST_TMPDIR", "temp"),
                                "bayesian_nn/input_data"),
        help="Directory to put the input data."
    )

    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.encoder_layers = [int(units) for units in FLAGS.encoder_layers.split(",")]
    FLAGS.activation = tf.nn.__getattribute__(FLAGS.activation)

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)