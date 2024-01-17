import jax
import jax.numpy as jnp
import numpy as np
from numpyro.examples.datasets import LYNXHARE, load_dataset
from utils_lotka import post_samples

print(f'JAX host: {jax.process_index()} / {jax.process_count()}')
print(f'JAX local devices: {jax.local_devices()}')

# Get data from numpyro

_, fetch = load_dataset(LYNXHARE, shuffle=False)
year, data = fetch()

# Run model to get different posteriors for different prior
# Baseline prior
num_results = 4000
num_burnin_steps = 1000
key = 1

# Baseline prior (same as in the Stan example)

ref = post_samples(num_results, num_burnin_steps,
                   key, u_init=jnp.log(jnp.log(jnp.array([10, 10.0]))))

# Piror1
b1 = post_samples(num_results, num_burnin_steps,
                  key, u_init=jnp.log(jnp.array([2.0, 2.0])))

# Piror2 
a = post_samples(num_results, num_burnin_steps, key, u_init=jnp.log(
    jnp.array([15, 6])), sigma1=jnp.array([2.0, 0.2]))

# Piror3
b2 = post_samples(num_results, num_burnin_steps, key, u_init=jnp.log(
    jnp.array([15, 6])), sigma1=jnp.array([1.0, 0.1]))

# Piror4
b3 = post_samples(num_results, num_burnin_steps, key, u_init=jnp.log(
    jnp.array([10, 10])), sigma1=jnp.array([1.0, 0.2]))


# Get reults in form for ott library
prefd = np.column_stack((jnp.squeeze(ref.theta), ref.z_init, ref.sigma))
pd1 = np.column_stack((jnp.squeeze(b1.theta), b1.z_init, b1.sigma))
pd2 = np.column_stack((jnp.squeeze(a.theta), a.z_init, a.sigma))
pd3 = np.column_stack((jnp.squeeze(b2.theta), b2.z_init, b2.sigma))
pd4 = np.column_stack((jnp.squeeze(b3.theta), b3.z_init, b3.sigma))

# Save posteriors
jnp.save('./prefd', prefd, allow_pickle=True)
jnp.save('./pd1', pd1, allow_pickle=True)
jnp.save('./pd2', pd2, allow_pickle=True)
jnp.save('./pd3', pd3, allow_pickle=True)
jnp.save('./pd4', pd4, allow_pickle=True)


