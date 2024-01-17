from numpyro.examples.datasets import LYNXHARE, load_dataset
from jax import random
from jax.experimental.ode import odeint
import jax.numpy as jnp
from tensorflow_probability.python.internal.backend.jax.compat import v2 as tf
import tensorflow_probability.substrates.jax as tfp
from utils_summary import posterior_summary

from jax.config import config
config.update("jax_enable_x64", True)

tfd = tfp.distributions
tfb = tfp.bijectors

# Get data from numpyro
_, fetch = load_dataset(LYNXHARE, shuffle=False)
year, data = fetch()


def dz_dt(z, t, theta):
    """
    Lotka–Volterra equations. Real positive parameters alpha, beta, gamma, delta
    describes the interaction of two species.
    """
    u, v = z
    alpha, beta, gamma, delta = (
        theta[..., 0], theta[..., 1], theta[..., 2], theta[..., 3])
    du_dt = (alpha - beta * v) * u
    dv_dt = (-gamma + delta * u) * v
    return jnp.stack([du_dt, dv_dt])


def solve(theta, z_init):
    """Solve for populations at finite set of times
    """
    t_obs = jnp.arange(float(data.shape[0]))

    results = odeint(
        dz_dt,
        z_init,
        t_obs,
        theta,
        rtol=1e-6,
        atol=1e-5,
        mxstep=1000)
    return results


def joint_pos(u_init=jnp.log(
        jnp.array([10, 10.0])), sigma1=jnp.array([-1, 1])):
    """Gives the joint posterior as a function

    Args:
        u_init (float, optional): array of the initial conditions. Defaults to jnp.log(jnp.array([10, 1.0])).
    """
    def model():
        theta = yield tfd.JointDistributionCoroutine.Root(tfd.Sample(tfd.TruncatedNormal(loc=tf.cast([1.0, 0.05, 1.0, 0.05], dtype='float64'),
                                                                                         scale=tf.cast(
                                                                                         [[0.5, 0.05, 0.5, 0.05]], dtype='float64'),
                                                                                         low=tf.cast(
                                                                                         0.0, dtype='float64'),
                                                                                         high=tf.cast(10, dtype='float64')), name='theta'))
        z_init = yield tfd.JointDistributionCoroutine.Root(tfd.Sample(tfd.LogNormal(loc=tf.cast(u_init, dtype='float64'),
                                                                                    scale=tf.cast(1.0, dtype='float64')), name='z_init'))
        sigma = yield tfd.JointDistributionCoroutine.Root(tfd.Sample(tfd.LogNormal(loc=tf.cast(sigma1[0], dtype='float64'),
                                                                                   scale=tf.cast(sigma1[1], dtype='float64')),
                                                                     sample_shape=(2), name='sigma'))
        yield tfd.Independent(tfd.LogNormal(loc=jnp.log(solve(theta, z_init)),
                                            scale=sigma), reinterpreted_batch_ndims=1, name="y")
    return tfd.JointDistributionCoroutineAutoBatched(model)


def post_samples(num_results=3000, num_burnin_steps=1000, key=1, u_init=jnp.log(
        jnp.array([10, 10.0])), sigma1=jnp.array([-1, 1])):
    """ function to get posterior samples

    Args:
        num_results (int, optional): number of poterior samples. Defaults to 3000.
        num_burnin_steps (int, optional): number of burnin-samples. Defaults to 1000.
        key (int, optional): number used to genrate random numbers. Defaults to 1.
        u_init (float, optional): array of initial conditions. Defaults to jnp.log(jnp.array([10, 1.0])).

    Returns:
        StructTuple: Posterior samples
    """
    key = random.PRNGKey(key)
    key, subkey = random.split(key)
    dist = joint_pos(u_init, sigma1)
    posterior = dist.experimental_pin(y=data)
    bij = posterior.experimental_default_event_space_bijector()
    sampler = tfp.mcmc.TransformedTransitionKernel(
        tfp.experimental.mcmc.PreconditionedNoUTurnSampler(
            target_log_prob_fn=posterior.unnormalized_log_prob,
            step_size=0.001),
        bijector=bij)
    adaptive_sampler = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=sampler,
        num_adaptation_steps=int(0.8 * num_burnin_steps),
        target_accept_prob=0.75)

    current_state = posterior.sample_unpinned(seed=subkey)

    def run_chain(key):
        """ Posterior samples
        """
        return tfp.mcmc.sample_chain(
            num_results=num_results,
            current_state=current_state,
            kernel=adaptive_sampler,
            num_burnin_steps=num_burnin_steps,
            trace_fn=lambda cs, kr: kr,
            seed=subkey)

    # Run the chain
    pos_samples, kernel_results = run_chain(key)

    print(
        "Acceptance rate:",
        kernel_results.inner_results.inner_results.is_accepted.mean())
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

    lower = jnp.take_along_axis(
        sorted_samples, lower_idx[None, ...], axis=axis)
    upper = jnp.take_along_axis(
        sorted_samples, upper_idx[None, ...], axis=axis)

    return lower, upper
