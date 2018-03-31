# coding: utf-8

import numpy as np 
import tensorflow as tf 
import tensorflow_probability as tfp 
import matplotlib.pyplot as plt 

import warnings
warnings.filterwarnings('ignore')

tfd = tf.contrib.distributions 
tfe = tf.contrib.eager 

def make_training_data(num_samples, dims, sigma):
    dt = np.asarray(sigma).dtype
    zeros = tf.zeros(dims, dtype=dt)
    x = tfd.MultivariateNormalDiag(
        loc=zeros).sample(num_samples, seed=1)
    w = tfd.MultivariateNormalDiag(
        loc=zeros,scale_identity_multiplier=sigma).sample(seed=2)
    noise = tfd.Normal(
        loc=dt.type(0),
        scale=dt.type(1)).sample(num_samples,seed=3)

    y = tf.tensordot(x, w, axes=[[1], [0]]) + noise 

    return y, x, w 

def make_prior(sigma, dims):
    return tfd.MultivariateNormalDiag(loc=tf.zeros([dims], dtype=sigma.dtype),
            scale_identity_multiplier=sigma)

def make_likelihood(x, w):
    return tfd.MultivariateNormalDiag(loc=tf.tensordot(x, w, axes=[[1], [0]]))


# set up assumptions 
dtype = np.float32 
num_samples = 150
dims = 10 
num_iters = int(5e3)

true_sigma = dtype(0.3)
y, x, true_weights = make_training_data(num_samples, dims, true_sigma)

# estimate of params
log_sigma = tf.get_variable(name="log_sigma", initializer=dtype(0))
sigma = tf.exp(log_sigma)

weights = tf.get_variable(
    name="weights",
    initializer=np.random.randn(dims).astype(dtype),
    trainable=False
)


prior = make_prior(sigma, dims)

def joint_log_prob(w):
    return prior.log_prob(w) + make_likelihood(x, w).log_prob(y)

hmc = tfp.mcmc.HamiltonianMonteCarlo(target_log_prob_fn=joint_log_prob,
                                step_size=0.1,num_leapfrog_steps=5)

weights_update = weights.assign(hmc.one_step(weights,
                        hmc.bootstrap_results(weights))[0])

with tf.control_dependencies([weights_update]):
    loss = -prior.log_prob(weights)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
log_sigma_updates = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sigma_history = np.zeros(num_iters, dtype)
weights_history = np.zeros([num_iters, dims], dtype)

with tf.Session() as sess:
    sess.run(init)
    for i in range(num_iters):
        _, sigma_, weights_ = sess.run([log_sigma_updates, sigma, weights])
        weights_history[i, :] = weights_
        sigma_history[i] = sigma_ 
    true_weights_ = sess.run(true_weights)

# plot 

plt.plot(sigma_history)
plt.ylabel("sigma")
plt.xlabel("iteration")
plt.show()

print("mean error:", np.abs(np.mean(sigma_history) - true_sigma))