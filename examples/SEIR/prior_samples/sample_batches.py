import numpy as np
from jax import random


def run_refd_prior_multiple_times(dist, num_samples=5, num_iterations=3):
    """Samples from the prior faster since it draws fewer samples many times and combines all the samples

    Args:
        dist (_type_): Prior distribution to sample from
        num_samples (int, optional):  Defaults to 5.
        num_iterations (int, optional): Defaults to 3.

    Returns:
        _type_: array of samples
    """
    samples = []

    for i in range(num_iterations):
        # Use a different seed for each iteration
        key = random.PRNGKey(i)
        key, subkey = random.split(key)
        # Sample from prior
        refd_prior = dist.sample(num_samples, seed=subkey)
        Prior_samples = np.column_stack(
            (refd_prior.theta,
             refd_prior.p_reported,
             refd_prior.alpha))
        # Save the results
        samples.append(Prior_samples)
    samples = np.concatenate(samples, axis=0)
    return samples
