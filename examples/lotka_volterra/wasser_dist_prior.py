import ott
import numpy as np
from numpyro.examples.datasets import LYNXHARE, load_dataset
from jax import random
from jax.experimental.ode import odeint
import jax.numpy as jnp
from tensorflow_probability.python.internal.backend.jax.compat import v2 as tf
import tensorflow_probability.substrates.jax as tfp
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
    #jax.debug.print("{thetashape}", thetashape=theta.shape)
    # print(theta.shape)
    results = odeint(
        dz_dt,
        z_init,
        t_obs,
        theta,
        rtol=1e-6,
        atol=1e-5,
        mxstep=1000)
    return results

# The five  priiors used
# first prior


def model():
    theta = yield tfd.JointDistributionCoroutine.Root(tfd.Sample(tfd.TruncatedNormal(loc=tf.cast([1.0, 0.05, 1.0, 0.05], dtype='float64'),
                                                                                     scale=tf.cast(
                                                                                         [0.5, 0.05, 0.5, 0.05], dtype='float64'),
                                                                                     low=tf.cast(
                                                                                         0.0, dtype='float64'),
                                                                                     high=tf.cast(10, dtype='float64')), name='theta'))
    z_init = yield tfd.JointDistributionCoroutine.Root(tfd.Sample(tfd.LogNormal(loc=tf.cast(jnp.log(jnp.array([10, 10])), dtype='float64'),
                                                                                scale=tf.cast(1.0, dtype='float64')), name='z_init'))
    sigma = yield tfd.JointDistributionCoroutine.Root(tfd.Sample(tfd.LogNormal(loc=tf.cast(-1.0, dtype='float64'),
                                                                               scale=tf.cast(1, dtype='float64')),
                                                                 sample_shape=(2), name='sigma'))
    yield tfd.Independent(tfd.LogNormal(loc=jnp.log(solve(theta, z_init)),
                          scale=sigma), reinterpreted_batch_ndims=1, name="y")


dist = tfd.JointDistributionCoroutineAutoBatched(model)

key = random.PRNGKey(1)
key, subkey = random.split(key)
num_samples = 4000

# sample from prior

refd_prior = dist.sample(num_samples, seed=key)

# save the prior in the form that can be used for Wasserstein distance
# calculation by Ott

refd = np.column_stack((refd_prior.theta, refd_prior.z_init, refd_prior.sigma))

# save refd
jnp.save('./refd', refd, allow_pickle=True)

# Second prior

def model():
    theta = yield tfd.JointDistributionCoroutine.Root(tfd.Sample(tfd.TruncatedNormal(loc=tf.cast([1.0, 0.05, 1.0, 0.05], dtype='float64'),
                                                                                     scale=tf.cast(
                                                                                         [0.5, 0.05, 0.5, 0.05], dtype='float64'),
                                                                                     low=tf.cast(
                                                                                         0.0, dtype='float64'),
                                                                                     high=tf.cast(10, dtype='float64')), name='theta'))
    z_init = yield tfd.JointDistributionCoroutine.Root(tfd.Sample(tfd.LogNormal(loc=tf.cast(jnp.log(jnp.array([2.0, 2.0])), dtype='float64'),
                                                                                scale=tf.cast(1.0, dtype='float64')), name='z_init'))
    sigma = yield tfd.JointDistributionCoroutine.Root(tfd.Sample(tfd.LogNormal(loc=tf.cast(-1.0, dtype='float64'),
                                                                               scale=tf.cast(1, dtype='float64')),
                                                                 sample_shape=(2), name='sigma'))
    yield tfd.Independent(tfd.LogNormal(loc=jnp.log(solve(theta, z_init)),
                          scale=sigma), reinterpreted_batch_ndims=1, name="y")


dist1 = tfd.JointDistributionCoroutineAutoBatched(model)

# sample from prior

d1_prior = dist1.sample(num_samples, seed=key)

# save the prior in the form that can be used for Wasserstein distance
# calculation by Ott

d1 = np.column_stack((d1_prior.theta, d1_prior.z_init, d1_prior.sigma))
jnp.save('./d1', d1, allow_pickle=True)

# third prior


def model():
    theta = yield tfd.JointDistributionCoroutine.Root(tfd.Sample(tfd.TruncatedNormal(loc=tf.cast([1.0, 0.05, 1.0, 0.05], dtype='float64'),
                                                                                     scale=tf.cast(
                                                                                         [0.5, 0.05, 0.5, 0.05], dtype='float64'),
                                                                                     low=tf.cast(
                                                                                         0.0, dtype='float64'),
                                                                                     high=tf.cast(10, dtype='float64')), name='theta'))
    z_init = yield tfd.JointDistributionCoroutine.Root(tfd.Sample(tfd.LogNormal(loc=tf.cast(jnp.log(jnp.array([15, 6.])), dtype='float64'),
                                                                                scale=tf.cast(1.0, dtype='float64')), name='z_init'))
    sigma = yield tfd.JointDistributionCoroutine.Root(tfd.Sample(tfd.LogNormal(loc=tf.cast(2.0, dtype='float64'),
                                                                               scale=tf.cast(0.2, dtype='float64')),
                                                                 sample_shape=(2), name='sigma'))
    yield tfd.Independent(tfd.LogNormal(loc=jnp.log(solve(theta, z_init)),
                          scale=sigma), reinterpreted_batch_ndims=1, name="y")


dist2 = tfd.JointDistributionCoroutineAutoBatched(model)

# sample from prior

d2_prior = dist2.sample(num_samples, seed=key)

# save the prior in the form that can be used for Wasserstein distance
# calculation by Ott
d2 = np.column_stack((d2_prior.theta, d2_prior.z_init, d2_prior.sigma))
jnp.save('./d2', d2, allow_pickle=True)

# fourth  prior

def model():
    theta = yield tfd.JointDistributionCoroutine.Root(tfd.Sample(tfd.TruncatedNormal(loc=tf.cast([1.0, 0.05, 1.0, 0.05], dtype='float64'),
                                                                                     scale=tf.cast(
                                                                                         [0.5, 0.05, 0.5, 0.05], dtype='float64'),
                                                                                     low=tf.cast(
                                                                                         0.0, dtype='float64'),
                                                                                     high=tf.cast(10, dtype='float64')), name='theta'))
    z_init = yield tfd.JointDistributionCoroutine.Root(tfd.Sample(tfd.LogNormal(loc=tf.cast(jnp.log(jnp.array([15, 6.])), dtype='float64'),
                                                                                scale=tf.cast(1.0, dtype='float64')), name='z_init'))
    sigma = yield tfd.JointDistributionCoroutine.Root(tfd.Sample(tfd.LogNormal(loc=tf.cast(1.0, dtype='float64'),
                                                                               scale=tf.cast(0.1, dtype='float64')),
                                                                 sample_shape=(2), name='sigma'))
    yield tfd.Independent(tfd.LogNormal(loc=jnp.log(solve(theta, z_init)),
                          scale=sigma), reinterpreted_batch_ndims=1, name="y")


dist3 = tfd.JointDistributionCoroutineAutoBatched(model)

# sample from prior

d3_prior = dist3.sample(num_samples, seed=key)

# save the prior in the form that can be used for Wasserstein distance
# calculation by Ott
d3 = np.column_stack((d3_prior.theta, d3_prior.z_init, d3_prior.sigma))
jnp.save('./d3', d3, allow_pickle=True)

###
# Fifth prior

def model():
    theta = yield tfd.JointDistributionCoroutine.Root(tfd.Sample(tfd.TruncatedNormal(loc=tf.cast([1.0, 0.05, 1.0, 0.05], dtype='float64'),
                                                                                     scale=tf.cast(
                                                                                         [0.5, 0.05, 0.5, 0.05], dtype='float64'),
                                                                                     low=tf.cast(
                                                                                         0.0, dtype='float64'),
                                                                                     high=tf.cast(10, dtype='float64')), name='theta'))
    z_init = yield tfd.JointDistributionCoroutine.Root(tfd.Sample(tfd.LogNormal(loc=tf.cast(jnp.log(jnp.array([10, 10])), dtype='float64'),
                                                                                scale=tf.cast(1.0, dtype='float64')), name='z_init'))
    sigma = yield tfd.JointDistributionCoroutine.Root(tfd.Sample(tfd.LogNormal(loc=tf.cast(1.0, dtype='float64'),
                                                                               scale=tf.cast(0.2, dtype='float64')),
                                                                 sample_shape=(2), name='sigma'))
    yield tfd.Independent(tfd.LogNormal(loc=jnp.log(solve(theta, z_init)),
                          scale=sigma), reinterpreted_batch_ndims=1, name="y")


dist4 = tfd.JointDistributionCoroutineAutoBatched(model)

# sample from prior

d4_prior = dist4.sample(num_samples, seed=key)

# save the prior in the form that can be used for Wasserstein distance
# calculation by Ott
d4 = np.column_stack((d4_prior.theta, d4_prior.z_init, d4_prior.sigma))
jnp.save('./d4', d4, allow_pickle=True)

def compute_optimal_transport(a1, a2):
    """Computes the Wasserstein distance between two distributions (a1 and a2)in more than one dimension

    Args:
        a1 (nxm array): sample from a distribution or posterior samples
        a2 (nxm array): sample from a distribution or posterior samples

    Returns:
        float: the entropu regularized Wasserstein distance and the distance without rgularisation
    """
    cost_fn = ott.geometry.costs.SqEuclidean()
    geometry = ott.geometry.pointcloud.PointCloud(a1, a2, cost_fn=cost_fn)
    problem = ott.problems.linear.linear_problem.LinearProblem(geometry)

    solver = ott.solvers.linear.sinkhorn.Sinkhorn()
    ot_solution = solver(problem)

    print("Entropy regularized OT cost:",
          jnp.sqrt(ot_solution.reg_ot_cost))

    return print(
        "OT cost without entropy:",
        jnp.sqrt(
            jnp.sum(
                ot_solution.matrix *
                ot_solution.geom.cost_matrix))), jnp.sqrt(ot_solution.reg_ot_cost)

# Call funtion to get the results
# Compute distance between baseline prior and the other priors
wp0p1 = compute_optimal_transport(refd, d1)[1]
wp0p2 = compute_optimal_transport(refd, d2)[1]
wp0p3 = compute_optimal_transport(refd, d3)[1]
wp0p4 = compute_optimal_transport(refd, d4)[1]


print(f'The Wasserstein distance between baseline prior and p1 is: {wp0p1}')
print(f'The Wasserstein distance between baseline prior and p2 is: {wp0p2}')
print(f'The Wasserstein distance between baseline prior and p3 is: {wp0p3}')
print(f'The Wasserstein distance between baseline prior and p4 is: {wp0p4}')


# load saved posteriors
prefd = jnp.load('./prefd.npy', allow_pickle=True)
pd1 = jnp.load('./pd1.npy', allow_pickle=True)
pd2 = jnp.load('./pd2.npy', allow_pickle=True)
pd3 = jnp.load('./pd3.npy', allow_pickle=True)
pd4 = jnp.load('./pd4.npy', allow_pickle=True)


# compute distance between posteriors (WIM)
WIM01 = compute_optimal_transport(prefd, pd1)[1]
WIM02 = compute_optimal_transport(prefd, pd2)[1]
WIM03 = compute_optimal_transport(prefd, pd3)[1]
WIM04 = compute_optimal_transport(prefd, pd4)[1]


print(f"The WIM: \n")
print(f'\t The WIM between baseline posterior and p1 posterior  is: {WIM01}')
print(f'\t The WIM between baseline posterior and p2 posterior is: {WIM02}')
print(f'\t The WIM between baseline posterior and p3 posterior is: {WIM03}')
print(f'\t The WIM between baseline posterior and p4 posterior is: {WIM04}')


# Prior scaled WIM
WsIM01 = compute_optimal_transport(
    prefd, pd1)[1] / compute_optimal_transport(refd, d1)[1]
WsIM02 = compute_optimal_transport(
    prefd, pd2)[1] / compute_optimal_transport(refd, d2)[1]
WsIM03 = compute_optimal_transport(
    prefd, pd3)[1] / compute_optimal_transport(refd, d3)[1]
WsIM04 = compute_optimal_transport(
    prefd, pd4)[1] / compute_optimal_transport(refd, d4)[1]

print(f"The prior scaled WIM: \n")
print(
    f'\t The prior scaledWIM between baseline posterior and p1 posterior  is: {WsIM01}')
print(
    f'\t The prior scaledWIM between baseline posterior and p2 posterior is: {WsIM02}')
print(
    f'\t The prior scaledWIM between baseline posterior and p3 posterior is: {WsIM03}')
print(
    f'\t The prior scaledWIM between baseline posterior and p4 posterior is: {WsIM04}')

