# coding: utf-8

import numpy as np 
import functools 
import tensorflow as tf 
import tensorflow_probability as tfp 

tfd = tf.contrib.distribution 
tfe = tf.contrib.eager 

VI_KL_LOSSES = "vi_kl_losses"
VI_PRIORS = "vi_priors"
VI_QDISTS = "vi_qdists"

def make_posterior_mvndiag(true_getter, base_var, name=None):
    """
    Builds a diagonal Gaussian posterior for a Variable `base_var`.
    This method is intended to be passed as an argument to
    `build_bayesian_getter`, not called directly.
    By using MVNDiag, we double the number of parameters in the model.
    If this is a problem one can imagine using alternatives like:
    `MVNDiag(scale_identity=s)`.
    Args:
        true_getter: The true getter being wrapped by our custom
        Bayesian getter. This should be used in place of any
        `tf.get_variable()` calls inside this method.
        base_var: The `tf.Variable` for which we are defining a
        variational posterior. This is treated as the mean of
        the Gaussian approximate posterior.
        name: The `Variable`'s name as passed into the Bayesian
        getter (note this will include any prefixes added
        by the calling model's `variable_scope`).
    Returns:
        q: A Distribution object representing the posterior.
    """

    with tf.name_scope(name, "make_posterior_mvndiag", [base_var]):

        sigma_init = np.log(np.exp1m(0.01))
        scale_diag = tf.nn.softplus(
            true_getter(
                name="{}/make_posterior_mvndiag/softplus_inverse_sigma".format(name),
                shape=base_var.shape.as_list(),
                dtype=base_var.dtype,
                initializer=tf.constant_initializer(sigma_init),
                trainable=True
            )
        )

        return tfd.Independent