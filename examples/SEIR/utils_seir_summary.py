import pandas as pd
import jax.numpy as jnp


def posterior_summary(samples):
    theta = jnp.mean(samples.theta, axis=0)
    sd_theta = jnp.std(samples.theta, axis=0)
    p_reported = jnp.mean(samples.p_reported, axis=0)
    p_reported_sd = jnp.std(samples.p_reported, axis=0)
    invphi = round(jnp.mean(samples.alpha, axis=0), 3)
    sd_invphi = round(jnp.std(samples.alpha, axis=0), 3)

    data = {
        "": ["iota", "rho", "sigma", "Ihat", "Ehat", "lambda", "invphi"],
        "Posterior_mean": [
            round(theta[0], 3),
            round(theta[1], 3),
            round(theta[2], 3),
            round(theta[3], 3),
            round(theta[4], 3),
            p_reported,
            invphi,
        ],
        "standard_dev": [
            round(sd_theta[0], 3),
            round(sd_theta[1], 3),
            round(sd_theta[2], 3),
            round(sd_theta[3], 3),
            round(sd_theta[4], 3),
            p_reported_sd,
            sd_invphi,
        ],
    }

    df = pd.DataFrame(data)
    return df
