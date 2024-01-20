import pandas as pd
import ott
import numpy as np
from jax import random
from jax.experimental.ode import odeint
import jax.numpy as jnp
from tensorflow_probability.python.internal.backend.jax.compat import v2 as tf
import tensorflow_probability.substrates.jax as tfp
from jax.config import config
config.update("jax_enable_x64", True)

tfd = tfp.distributions
tfb = tfp.bijectors


prefd = jnp.load('./prefd.npy', allow_pickle=True)
pd1 = jnp.load('./pd1.npy', allow_pickle=True)
pd2 = jnp.load('./pd2.npy', allow_pickle=True)
pd3 = jnp.load('./pd3.npy', allow_pickle=True)
pd41 = jnp.load('./pd41.npy', allow_pickle=True)

refd = jnp.load('./refd.npy', allow_pickle=True)
d1 = jnp.load('./d1.npy', allow_pickle=True)
d2 = jnp.load('./d2.npy', allow_pickle=True)
d3 = jnp.load('./d3.npy', allow_pickle=True)
d4 = jnp.load('./d41.npy', allow_pickle=True)

def compute_optimal_transport(a1, a2):
    cost_fn = ott.geometry.costs.SqEuclidean()
    geometry = ott.geometry.pointcloud.PointCloud(a1, a2, cost_fn=cost_fn)
    problem = ott.problems.linear.linear_problem.LinearProblem(geometry)

    solver = ott.solvers.linear.sinkhorn.Sinkhorn()
    ot_solution = solver(problem)

    print("Entropy regularized OT cost:",
          jnp.sqrt(ot_solution.reg_ot_cost))

    print(
        "OT cost without entropy:",
        jnp.sqrt(
            jnp.sum(
                ot_solution.matrix *
                ot_solution.geom.cost_matrix)))

    return jnp.sqrt(ot_solution.reg_ot_cost)


# Call funtion to get the results
# Compute distance between baseline prior and the other priors

print('between baseline prior and the other priors')
compute_optimal_transport(refd, d1)
compute_optimal_transport(refd, d2)
compute_optimal_transport(refd, d3)
compute_optimal_transport(refd, d4)

# Compute distance between baseline posterior and other posterior
print('\n between baseline posterior and the other posteriors')
compute_optimal_transport(prefd, pd1)
compute_optimal_transport(prefd, pd2)
compute_optimal_transport(prefd, pd3)
compute_optimal_transport(prefd, pd41)

# Compute distance between posteriors and reference  priors
print('\n between baseline prior  and the other posteriors')
compute_optimal_transport(refd, prefd)
compute_optimal_transport(refd, pd1)
compute_optimal_transport(refd, pd2)
compute_optimal_transport(refd, pd3)
compute_optimal_transport(refd, d4)

# Compute distance between reference posteriors and priors
print('\n between baseline posterior and the other priors')
compute_optimal_transport(refd, prefd)
compute_optimal_transport(prefd, d1)
compute_optimal_transport(prefd, d2)
compute_optimal_transport(prefd, d3)
compute_optimal_transport(prefd, d4)

# Prior Normalised WIM
print('\n prior Normalised WIM')

ws42 = compute_optimal_transport(prefd, pd1)/compute_optimal_transport(refd, d1)
print(f'the prior normalised WIM for exp(42) is: {ws42}')

ws1 = compute_optimal_transport(prefd, pd2)/compute_optimal_transport(refd, d2)
print(f'the prior normalised WIM for exp(1) is: {ws1}')

wsgamma = compute_optimal_transport(prefd, pd3)/compute_optimal_transport(refd, d3)
print(f'the prior normalised WIM for gamma(16, 16) is: {wsgamma}')

ws150 = compute_optimal_transport(prefd, pd41)/compute_optimal_transport(refd, d4)
print(f'the prior normalised WIM for exp(150) is: {ws150}')
