import pandas as pd
import jax.numpy as jnp
from jax import random
from jax.experimental.ode import odeint
from tensorflow_probability.python.internal.backend.jax.compat import v2 as tf
from utils_seir_summary import posterior_summary

import tensorflow_probability.substrates.jax as tfp
from jax.config import config

config.update("jax_enable_x64", True)


tfd = tfp.distributions
tfb = tfp.bijectors


# import data

filtered_df = pd.read_csv("filtered_data.csv")


def seir(y, t, theta):
    N = filtered_df.population.values[0]

    beta, gamma, a, _i0, _e0 = (
        theta[..., 0],
        theta[..., 1],
        theta[..., 2],
        theta[..., 3],
        theta[..., 4],
    )

    dS_dt = -beta * y[2] * y[0] / N
    dE_dt = beta * y[2] * y[0] / N - a * y[1]
    dI_dt = a * y[1] - gamma * y[2]
    dR_dt = gamma * y[2]

    return jnp.stack([dS_dt, dE_dt, dI_dt, dR_dt])


def solve(theta):
    """Solve for populations at finite set of times"""
    N = filtered_df.population.values[0]  # Total number of individuals, N
    t_obs = jnp.arange(0, float(len(filtered_df.population.values)))
    i0 = theta[..., 3]
    e0 = theta[..., 4]
    x_init = jnp.stack([N - i0 - e0, e0, i0, 0])
    results = odeint(seir, x_init, t_obs, theta, rtol=1e-6, atol=1e-5, mxstep=1000)
    return results


def joint_pos(phi=5):
    """Gives the joint posterior as a function

    Args:
        phi (float, optional): overdispersion parameter of the negative binomial distribution
    """

    def model():
        theta = yield tfd.JointDistributionCoroutine.Root(
            tfd.Sample(
                tfd.TruncatedNormal(
                    loc=tf.cast([2, 0.6, 0.4, 0.0, 0.0], dtype="float64"),
                    scale=tf.cast([1.0, 0.5, 0.5, 1.0, 1.0], dtype="float64"),
                    low=tf.cast(0.0, dtype="float64"),
                    high=tf.cast(5.0, dtype="float64"),
                ),
                name="theta",
            )
        )

        p_reported = yield tfd.JointDistributionCoroutine.Root(
            tfd.Sample(
                tfd.Beta(
                    concentration0=tf.cast(1.0, dtype="float64"),
                    concentration1=tf.cast(2.0, dtype="float64"),
                ),
                name="p_reported",
            )
        )

        inv_alpha = yield tfd.JointDistributionCoroutine.Root(
            tfd.Sample(
                tfd.Exponential(rate=tf.cast(phi, dtype="float64")), name="alpha"
            )
        )

        alpha = 1 / inv_alpha

        mu = solve(theta)

        incidence = (mu[:-1, 1] - mu[1:, 1] + mu[:-1, 0] - mu[1:, 0]) * p_reported

        yield tfd.Independent(
            tfd.NegativeBinomial.experimental_from_mean_dispersion(
                mean=incidence, dispersion=alpha
            ),
            reinterpreted_batch_ndims=1,
            name="y",
        )

    return tfd.JointDistributionCoroutineAutoBatched(model)


def post_samples(num_results=3000, num_burnin_steps=1000, key=1, phi=5):
    """function to get posterior samples

    Args:
        num_results (int, optional): number of poterior samples. Defaults to 3000.
        num_burnin_steps (int, optional): number of burnin-samples. Defaults to 1000.
        key (int, optional): number used to genrate random numbers. Defaults to 1.
        phi (float, optional): overdispersion parameter of the negative binomial distribution

    Returns:
        StructTuple: Posterior samples
    """
    key = random.PRNGKey(key)
    key, subkey = random.split(key)
    dist = joint_pos(phi)
    posterior = dist.experimental_pin(y=filtered_df.new_cases.values[:-1])
    bij = posterior.experimental_default_event_space_bijector()

    sampler = tfp.mcmc.TransformedTransitionKernel(
        tfp.experimental.mcmc.PreconditionedNoUTurnSampler(
            target_log_prob_fn=posterior.unnormalized_log_prob, step_size=0.001
        ),
        bijector=bij,
    )

    adaptive_sampler = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=sampler,
        num_adaptation_steps=int(0.8 * num_burnin_steps),
        target_accept_prob=0.75,
    )

    current_state = posterior.sample_unpinned(seed=subkey)

    def run_chain(key):
        """Posterior samples"""
        return tfp.mcmc.sample_chain(
            num_results=num_results,
            current_state=current_state,
            kernel=adaptive_sampler,
            num_burnin_steps=num_burnin_steps,
            trace_fn=lambda cs, kr: kr,
            seed=key,
        )

    # Run the chain
    pos_samples, kernel_results = run_chain(key)

    print(
        "Acceptance rate:",
        kernel_results.inner_results.inner_results.is_accepted.mean(),
    )

    print(posterior_summary(pos_samples))

    return pos_samples


def hdi(pos_samples, prob=0.95, axis=None):
    """Get the highest density interval (HDI) with a given probability.
    Use `jnp.squeeze()` on posterior samples if necessary.

    Args:
      pos_samples: A JAX NumPy array of posterior samples.
      prob: The desired confidence level.
      axis: The axis along which to calculate the HDI. If `None`, the HDI is computed for the flattened array.

    Returns:
      A tuple of the lower and upper bounds of the HDI.
    """

    alpha = 1 - prob
    sorted_samples = jnp.sort(pos_samples, axis=axis)
    cdf = jnp.cumsum(sorted_samples, axis=axis) / sorted_samples.shape[axis]

    lower_idx = jnp.argmin(jnp.abs(cdf - alpha / 2), axis=axis)
    upper_idx = jnp.argmin(jnp.abs(cdf - (1 - alpha / 2)), axis=axis)

    lower = jnp.take_along_axis(sorted_samples, lower_idx[None, ...], axis=axis)
    upper = jnp.take_along_axis(sorted_samples, upper_idx[None, ...], axis=axis)

    return lower, upper


def joint_pos1(a=5):
    """Gives the joint posterior as a function

    Args:
        phi (float, optional): overdispersion parameter of the negative binomial distribution
    """

    def model():
        theta = yield tfd.JointDistributionCoroutine.Root(
            tfd.Sample(
                tfd.TruncatedNormal(
                    loc=tf.cast([2, 0.6, 0.4, 0.0, 0.0], dtype="float64"),
                    scale=tf.cast([1.0, 0.5, 0.5, 1.0, 1.0], dtype="float64"),
                    low=tf.cast(0.0, dtype="float64"),
                    high=tf.cast(5.0, dtype="float64"),
                ),
                name="theta",
            )
        )

        p_reported = yield tfd.JointDistributionCoroutine.Root(
            tfd.Sample(
                tfd.Beta(
                    concentration0=tf.cast(1.0, dtype="float64"),
                    concentration1=tf.cast(2.0, dtype="float64"),
                ),
                name="p_reported",
            )
        )

        inv_alpha = yield tfd.JointDistributionCoroutine.Root(
            tfd.Sample(
                tfd.Gamma(
                    concentration=tf.cast(a, dtype="float64"),
                    rate=tf.cast(a, dtype="float64"),
                ),
                name="alpha",
            )
        )

        alpha = 1 / inv_alpha

        mu = solve(theta)

        incidence = (mu[:-1, 1] - mu[1:, 1] + mu[:-1, 0] - mu[1:, 0]) * p_reported

        yield tfd.Independent(
            tfd.NegativeBinomial.experimental_from_mean_dispersion(
                mean=incidence, dispersion=alpha
            ),
            reinterpreted_batch_ndims=1,
            name="y",
        )

    return tfd.JointDistributionCoroutineAutoBatched(model)


def post_samples1(num_results=3000, num_burnin_steps=1000, key=1, a=5):
    """function to get posterior samples

    Args:
        num_results (int, optional): number of poterior samples. Defaults to 3000.
        num_burnin_steps (int, optional): number of burnin-samples. Defaults to 1000.
        key (int, optional): number used to genrate random numbers. Defaults to 1.
        phi (float, optional): overdispersion parameter of the negative binomial distribution

    Returns:
        StructTuple: Posterior samples
    """
    key = random.PRNGKey(key)
    key, subkey = random.split(key)
    dist = joint_pos1(a)
    posterior = dist.experimental_pin(y=filtered_df.new_cases.values[:-1])
    bij = posterior.experimental_default_event_space_bijector()

    sampler = tfp.mcmc.TransformedTransitionKernel(
        tfp.experimental.mcmc.PreconditionedNoUTurnSampler(
            target_log_prob_fn=posterior.unnormalized_log_prob, step_size=0.001
        ),
        bijector=bij,
    )

    adaptive_sampler = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=sampler,
        num_adaptation_steps=int(0.8 * num_burnin_steps),
        target_accept_prob=0.75,
    )

    current_state = posterior.sample_unpinned(seed=subkey)

    def run_chain(key):
        """Posterior samples"""
        return tfp.mcmc.sample_chain(
            num_results=num_results,
            current_state=current_state,
            kernel=adaptive_sampler,
            num_burnin_steps=num_burnin_steps,
            trace_fn=lambda cs, kr: kr,
            seed=key,
        )

    # Run the chain
    pos_samples, kernel_results = run_chain(key)

    print(
        "Acceptance rate:",
        kernel_results.inner_results.inner_results.is_accepted.mean(),
    )

    print(posterior_summary(pos_samples))

    return pos_samples
