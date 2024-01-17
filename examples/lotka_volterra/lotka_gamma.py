from jax import random
from jax.experimental.ode import odeint
import jax.numpy as jnp
from tensorflow_probability.python.internal.backend.jax.compat import v2 as tf
import tensorflow_probability.substrates.jax as tfp
from jax.config import config
config.update("jax_enable_x64", True)


tfd = tfp.distributions
tfb = tfp.bijectors


def posterior_samples1(key, data, num_results=2000, num_burnin_steps=1000, mu=[
        [1.0, 0.05, 1.0, 0.05], [0.5, 0.05, 0.5, 0.05]], z_init_mu=jnp.array([10, 1]), sigma_mu=[1.0, 1.0]):

    def dz_dt(z, t, theta):
        """
        Lotka–Volterra equations. Real positive parameters `alpha`, `beta`, `gamma`, `delta`
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

    def unnorm_pos(mu, z_init_mu, sigma_mu, num_results, num_burnin_steps):
        def model():
            theta = yield tfd.JointDistributionCoroutine.Root(tfd.Sample(tfd.TruncatedNormal(loc=tf.cast(mu[0], dtype='float64'),
                                                                                             scale=tf.cast(mu[1], dtype='float64'),
                                                                                             low=tf.cast(0.0, dtype='float64'),
                                                                                             high=tf.cast(10, dtype='float64')), name='theta'))
            z_init = yield tfd.JointDistributionCoroutine.Root(tfd.Sample(tfd.LogNormal(loc=tf.cast(jnp.log(jnp.log(z_init_mu)), dtype='float64'),
                                                                                        scale=tf.cast(1.0, dtype='float64')), name='z_init'))
            sigma = yield tfd.JointDistributionCoroutine.Root(tfd.Sample(tfd.InverseGamma(concentration=tf.cast(sigma_mu[0], dtype='float64'),
                                                                                          scale=tf.cast(sigma_mu[1], dtype='float64')),
                                                                         sample_shape=(2), name='sigma'))
            yield tfd.Independent(tfd.LogNormal(loc=jnp.log(solve(theta, z_init)),
                                                scale=sigma), reinterpreted_batch_ndims=1, name="y")
        return tfd.JointDistributionCoroutineAutoBatched(model)

    #seed = SystemRandom().randint(np.iinfo(np.uint32).min, np.iinfo(np.uint32).max)
    key1 = random.PRNGKey(key)
    key, subkey = random.split(key1)

    pos = unnorm_pos(mu, z_init_mu, sigma_mu, num_results, num_burnin_steps)
    posterior = pos.experimental_pin(y=data)
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
            trace_fn=None,
            seed=subkey)

    pos_samples = run_chain(key)

    return pos_samples
