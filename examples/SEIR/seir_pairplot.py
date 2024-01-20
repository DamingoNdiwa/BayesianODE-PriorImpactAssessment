import jax
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots
from jax import random
from jax.random import PRNGKey
import jax.numpy as jnp
from jax.config import config
import matplotlib.pyplot as plt
from utils_seir import joint_pos, post_samples, hdi, joint_pos1, post_samples1

config.update("jax_enable_x64", True)
plt.style.use(['science', 'ieee'])


# import data 
df =pd.read_csv("filtered_data.csv")


# Run model and save posterior samples results.

num_results = 3000
num_burnin_steps = 1000
key = 1

pos_samples1 = post_samples(num_results, num_burnin_steps,
                            key, phi=42)

dist = joint_pos()
keys = random.split(PRNGKey(1), num_results)

# posterior predictive check

posterior_predictive1 = jax.vmap(
    lambda key,
    sample: dist.sample(
        seed=key,
        value=sample))(
            keys,
    pos_samples1)

# Get the posterior prdictive mean

mu1 = jnp.mean(posterior_predictive1.y, axis=0)
pi1 = jnp.percentile(posterior_predictive1.y, jnp.array([25, 75]), axis=0)

# Get posteriors in dataframe

post_save = pd.DataFrame(
    np.column_stack(
        (jnp.squeeze(
            pos_samples1.theta), jnp.squeeze(
                pos_samples1.p_reported), jnp.squeeze(
                    pos_samples1.alpha))))


# parameter names to used for plots in model m4
parameters = [
    r'$\eta$',
    r'$\rho$',
    r'$\sigma$',
    r'$\hat{I}$',
    r'$\hat{E}$',
    r'$\lambda$',
    r'$\phi$']

post_save.columns = parameters

fig = plt.figure()


def pairplot(postdata):
    """function to make pairplots for model m4 and save

    Args:
        postdata (data): posterior samples in a pandas datarame

    Returns:
        pair plots
    """
    sns.set_context("paper",
                    rc={'font.size': 35,
                        'legend.title_fontsize': 20,
                        'axes.labelsize': 35,
                        'xtick.labelsize': 27,
                        'ytick.labelsize': 27,
                        'xtick.major.size': 0.0,
                        'ytick.major.size': 0.0,
                        'ytick.minor.size': 0.0,
                        'patch.linewidth': 5.0,
                        'xtick.minor.size': 0.0})

    g = sns.PairGrid(postdata, diag_sharey=False,
                     corner=True, despine=True)
    g.map_lower(sns.kdeplot, color='blue', common_norm=False, fill=True)
    g.map_diag(sns.kdeplot, color='blue', ls='-', common_norm=False)
    fig.set_figwidth(0.2)
    fig.set_figheight(1.5)
    g.fig.autofmt_xdate(rotation=45)  # Rotate x-tick labels
    pairplot = g.figure.savefig(
        './marposseir1.pdf')

    return g


# call function to give pair plot
pairplot(post_save)

# Call the function to get the 95% CI with any parameter of your choice
# Get the 95% CI for the intial conditions
a = hdi(jnp.squeeze(pos_samples1.theta), prob=0.95, axis=0)
print(a)
