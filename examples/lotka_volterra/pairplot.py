from utils_lotka import hdi, post_samples
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numpyro.examples.datasets import LYNXHARE, load_dataset
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)
import scienceplots
plt.style.use(['science', 'ieee'])
# Get data from numpyro
_, fetch = load_dataset(LYNXHARE, shuffle=False)
year, data = fetch()

# Run model with three different priors for initial values and save the
# results.
# Stan prior
num_results = 3000
num_burnin_steps = 2000
key = 1

# Make pairplots of the margial posterior for one of the models
pos_samples0 = post_samples(num_results, num_burnin_steps,
                            key=1, u_init=jnp.log(jnp.array([2, 2])))

post_save = pd.DataFrame(
    np.column_stack(
        (jnp.squeeze(
            pos_samples0.theta), jnp.squeeze(
                pos_samples0.z_init), jnp.squeeze(
                    pos_samples0.sigma))))


# parameter names to used for plots in model m4
parameters = [
    r'$\alpha$',
    r'$\gamma$',
    r'$\delta$',
    r'$\beta$',
    r'$\hat{u}$',
    r'$\hat{v}$',
    r'$\sigma_{u}$',
    r'$\sigma_{v}$']

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
    fig.set_figwidth(0.5)
    fig.set_figheight(1.5)
    g.fig.autofmt_xdate(rotation=45)  # Rotate x-tick labels
    # plt.show()
    g.figure.savefig('./marpos.pdf')

    return g


# call function to give pair plot
pairplot(post_save)

# Call the function to get the 95% CI with any parameter of your choice
# Get the 95% CI for the intial conditions
a = hdi(jnp.squeeze(pos_samples0.z_init), prob=0.95, axis=0)
print(a)
