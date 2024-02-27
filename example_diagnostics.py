import numpy as np
import jax.numpy as jnp
from jax.config import config
from pymcmcstat.chain.ChainStatistics import geweke, integrated_autocorrelation_time
config.update("jax_enable_x64", True)

from utils_diagnostics import create_dataframe

# import data 
pefd = jnp.load('examples/SEIR/prefd.npy', allow_pickle=True)

post_samples = create_dataframe(pefd, model="seir")

# If the values are less than N/50, we can confirm the convergence of the
# parameter.
a = np.round(integrated_autocorrelation_time(post_samples.values), 3)
print(
    f'\nThe integrated autocorrelation time for parameters of Model is :\n\t{a[0]}')
b = np.round(geweke(post_samples.values, 0.2, 0.6), 3)
# For Geweke, if the p-value is > 0.05, we can confirm convergence
print(f'\nThe p-values of parameters for Model is :\n\t{b[1]}')

