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
from sample_batches import run_refd_prior_multiple_times

tfd = tfp.distributions
tfb = tfp.bijectors

# Import data
filtered_df = pd.read_csv("../filtered_data.csv")

def seir(y, t, theta):
    N = filtered_df.population.values[0]

    beta, gamma, a, i0, e0 = (
        theta[..., 0], theta[..., 1], theta[..., 2], theta[..., 3], theta[..., 4])

    dS_dt = -beta * y[2] * y[0] / N
    dE_dt = beta * y[2] * y[0] / N - a * y[1]
    dI_dt = a * y[1] - gamma * y[2]
    dR_dt = gamma * y[2]

    return jnp.stack([dS_dt, dE_dt, dI_dt, dR_dt])


def solve(theta):
    """Solve for populations at finite set of times
    """
    N = filtered_df.population.values[0]  # Total number of individuals, N
    t_obs = jnp.arange(0, float(len(filtered_df.population.values)))
    i0 = theta[..., 3]
    e0 = theta[..., 4]
    x_init = jnp.stack([N - i0 - e0, e0, i0, 0])
    results = odeint(
        seir,
        x_init,
        t_obs,
        theta,
        rtol=1e-6,
        atol=1e-5,
        mxstep=1000)
    return results


# fourth  prior

def model():
    theta = yield tfd.JointDistributionCoroutine.Root(
        tfd.Sample(
            tfd.TruncatedNormal(
                loc=tf.cast([2, 0.6, 0.4, 0.0, 0.0], dtype='float64'),
                scale=tf.cast([1.0, 0.5, 0.5, 1.0, 1.0], dtype='float64'),
                low=tf.cast(0.0, dtype='float64'),
                high=tf.cast(5.0, dtype='float64')), name='theta'))

    p_reported = yield tfd.JointDistributionCoroutine.Root(
        tfd.Sample(tfd.Beta(concentration0=tf.cast(1.0, dtype='float64'),
                            concentration1=tf.cast(2.0, dtype='float64')), name='p_reported'))

    inv_alpha = yield tfd.JointDistributionCoroutine.Root(tfd.Sample(
        tfd.Gamma(concentration=tf.cast(16, dtype='float64'),
                  rate=tf.cast(16, dtype='float64')), name='alpha'))

    alpha = 1 / inv_alpha

    mu = solve(theta)

    incidence = (mu[:-1, 1] - mu[1:, 1] +
                 mu[:-1, 0] - mu[1:, 0]) * p_reported

    yield tfd.Independent(tfd.NegativeBinomial.experimental_from_mean_dispersion(mean=incidence, dispersion=alpha), reinterpreted_batch_ndims=1, name="y")


dist3_prior = tfd.JointDistributionCoroutineAutoBatched(model)

# sample from prior
d31 = run_refd_prior_multiple_times(dist3_prior, num_samples=60, num_iterations=50)

jnp.save('../d31', d31, allow_pickle=True)
print('completed')
