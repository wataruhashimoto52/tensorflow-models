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

def main(argv):
    del argv 
