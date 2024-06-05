import pandas as pd
import jax.numpy as jnp


def posterior_summary(samples):
    theta = jnp.mean(samples.theta, axis=0)
    sd_theta = jnp.std(samples.theta, axis=0)
    init = round(jnp.mean(samples.z_init, axis=0), 3)
    sd_init = round(jnp.std(samples.z_init, axis=0), 3)
    sd = round(jnp.mean(samples.sigma, axis=0), 3)
    sd_sd = round(jnp.std(samples.sigma, axis=0), 3)

    data = {
        "": ["alpha", "beta", "delta", "gamma", "mu", "nu", "sd_mu", "sd_nu"],
        "Posterior_mean": [
            round(theta[0][0], 3),
            round(theta[0][1], 3),
            round(theta[0][2], 3),
            round(theta[0][3], 3),
            init[0],
            init[1],
            sd[0],
            sd[1],
        ],
        "standard_dev": [
            round(sd_theta[0][0], 3),
            round(sd_theta[0][1], 3),
            round(sd_theta[0][2], 3),
            round(sd_theta[0][3]),
            sd_init[0],
            sd_init[1],
            sd_sd[0],
            sd_sd[1],
        ],
    }

    df = pd.DataFrame(data)
    return df
