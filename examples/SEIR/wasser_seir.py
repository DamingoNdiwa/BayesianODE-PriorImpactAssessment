import jax
import jax.numpy as jnp
import ott
import numpy as np
from utils_seir import post_samples, post_samples1
import pandas as pd


print(f'JAX host: {jax.process_index()} / {jax.process_count()}')
print(f'JAX local devices: {jax.local_devices()}')

# Run model to get different posteriors for different prior
# Baseline prior

a = 20#3000
b = 20#1000

exref = post_samples(num_results=a, num_burnin_steps=b, key=1, phi=5)

# Piror1
ex42 = post_samples(num_results=a, num_burnin_steps=b, key=1, phi=42)

# prior2
ex1 = post_samples(num_results=a, num_burnin_steps=b, key=1, phi=1)

# prior3

gamma1 = post_samples1(num_results=a, num_burnin_steps=b, key=1, a=16)

# prior4

ex70 = post_samples(num_results=a, num_burnin_steps=b, key=1, phi=150)

# Get reults in form for ott library
prefd = np.column_stack((exref.theta, exref.p_reported, exref.alpha))
pd1 = np.column_stack((ex42.theta, ex42.p_reported, ex42.alpha))
pd2 = np.column_stack((ex1.theta, ex1.p_reported, ex1.alpha))
pd3 = np.column_stack((gamma1.theta, gamma1.p_reported, gamma1.alpha))
pd4 = np.column_stack((ex70.theta, ex70.p_reported, ex70.alpha))

# Save posteriors
jnp.save('./prefd', prefd, allow_pickle=True)
jnp.save('./pd1', pd1, allow_pickle=True)
jnp.save('./pd2', pd2, allow_pickle=True)
jnp.save('./pd3', pd3, allow_pickle=True)
jnp.save('./pd41', pd4, allow_pickle=True)
