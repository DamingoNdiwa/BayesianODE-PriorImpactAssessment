import jax
import matplotlib.pyplot as plt
from numpyro.examples.datasets import LYNXHARE, load_dataset
from jax import random
from jax.random import PRNGKey
import jax.numpy as jnp
from jax.config import config
from utils_lotka import joint_pos, post_samples
config.update("jax_enable_x64", True)
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

# Run function to get posterior smaples for three different sets of priors
# We change only the initial conditiosn
# First prior which is the stan prior

pos_samples0 = post_samples(num_results, num_burnin_steps,
                            key, u_init=jnp.log(jnp.array([10, 10.0])))

# Second prior
pos_samples1 = post_samples(num_results, num_burnin_steps,
                            key, u_init=jnp.log(jnp.array([2.0, 2.0])))

# Third prior

pos_samples2 = post_samples(num_results, num_burnin_steps, key, u_init=jnp.log(
    jnp.array([15, 6])), sigma1=jnp.array([2.0, 0.2]))

pos_samples3 = post_samples(num_results, num_burnin_steps, key, u_init=jnp.log(
    jnp.array([15, 6])), sigma1=jnp.array([1.0, 0.1]))

# Posterior predictive check
# for the first prior

dist = joint_pos()
keys = random.split(PRNGKey(1), num_results)
posterior_predictive0 = jax.vmap(
    lambda key,
    sample: dist.sample(
        seed=key,
        value=sample))(
            keys,
    pos_samples0)

# get the posterior pridictive mean
mu0 = jnp.mean(posterior_predictive0.y, axis=0)
pi0 = jnp.percentile(posterior_predictive0.y, jnp.array([25, 75]), axis=0)

# For the second prior
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

# For the third prior

posterior_predictive2 = jax.vmap(
    lambda key,
    sample: dist.sample(
        seed=key,
        value=sample))(
            keys,
    pos_samples2)

# Get the posterior pridictive mean
mu2 = jnp.mean(posterior_predictive2.y, axis=0)
pi2 = jnp.percentile(posterior_predictive2.y, jnp.array([25, 75]), axis=0)

posterior_predictive3 = jax.vmap(
    lambda key,
    sample: dist.sample(
        seed=key,
        value=sample))(
            keys,
    pos_samples3)

# Get the posterior pridictive mean
mu3 = jnp.mean(posterior_predictive3.y, axis=0)
pi3 = jnp.percentile(posterior_predictive3.y, jnp.array([25, 75]), axis=0)


# posterior predictive cheks for all models with prediction intervals
# separate the Hare from the Lynx in the plots

fig, axs = plt.subplots(2, 1, sharex=True)
axs[0].plot(year, mu0[:, 0], "k-", label=r"$p(\theta_0)$", lw=1)
axs[0].plot(year, mu1[:, 0], "r-", label=r"$p(\theta_1)$", lw=1)
axs[0].plot(year, mu2[:, 0], "b-", label=r"$p(\theta_2)$", lw=1)
axs[0].plot(year, mu3[:, 0], "y-", label=r"$p(\theta_3)$", lw=1)
axs[0].plot(year, data[:, 0], "ko", mfc="none", ms=2.5, label="observed hare")
axs[0].set_ylim(-1, 260)
yticks = jnp.arange(0, 280, 70)
axs[0].set_yticks(yticks)
axs[0].tick_params(axis='both', labelsize=12)
axs[0].legend(fontsize='small', ncol=2, bbox_to_anchor=(
    0.45, - 0.1, 0.56, 1.8), frameon=True, title='Hare', title_fontsize='x-small')
axs[1].plot(year, data[:, 1], "bx", ms=2.5, label="observed lynx")
axs[1].plot(year, mu0[:, 1], "k-.", label=r"$p(\theta_0)$", lw=1.2)
axs[1].plot(year, mu1[:, 1], "r-.", label=r"$p(\theta_1)$", lw=1)
axs[1].plot(year, mu2[:, 1], "b-.", label=r"$p(\theta_2)$", lw=1)
axs[1].plot(year, mu3[:, 1], "y-", label=r"$p(\theta_3)$", lw=1)

axs[1].set_ylim(0, 260)
yticks = jnp.arange(0, 260, 70)
axs[1].set_yticks(yticks)
axs[1].tick_params(axis='both', labelsize=12)
axs[1].set_ylabel('Pelts (thousands)', fontsize=12, y=1)
axs[1].yaxis.labelpad = 2
axs[1].set_xlabel('Year', fontsize=12)
axs[1].legend(fontsize='small', ncol=2, bbox_to_anchor=(
    0.45, -0.1, 0.55, 1.3), frameon=True, title='Lynx', title_fontsize='xx-small')
fig.set_figwidth(3.5)
fig.set_figheight(2.5)
plt.subplots_adjust(hspace=0.2)  # Increase the spacing between subplots
plt.savefig("./odesppc_012.pdf", dpi=1000)

# Make posterior predictive plots for all models
fig, axs = plt.subplots(3, 2, sharex=True, sharey=True)
axs[0, 0].plot(year, mu0[:, 0], "k-", label=r"$p(\theta_0)$", lw=1)
axs[0, 0].plot(year, mu1[:, 0], "r-", label=r"$p(\theta_1)$", lw=1)
axs[0, 0].plot(year, mu2[:, 0], "b-", label=r"$p(\theta_2)$", lw=1)
axs[0, 0].plot(year, mu3[:, 0], "y-", label=r"$p(\theta_3)$", lw=1)
axs[0, 0].plot(year, data[:, 0], "ko", mfc="none",
               ms=2.5, label="observed hare")
axs[0, 0].tick_params(axis='both', labelsize=12)
axs[0, 0].set_ylabel('Pelts (thousands)', fontsize=12)
axs[0, 0].legend(fontsize='small', ncol=2, bbox_to_anchor=(
    0.45, - 0.1, 0.56, 1.48), frameon=True, title='Hare', title_fontsize='small')
axs[0, 0].text(1825, 250, s='\\textbf{a}', fontsize=20)
axs[0, 1].plot(year, data[:, 1], "bx", ms=2.5, label="observed lynx")
axs[0, 1].plot(year, mu0[:, 1], "k-.", label=r"$p(\theta_0)$", lw=1.2)
axs[0, 1].plot(year, mu1[:, 1], "r-.", label=r"$p(\theta_1)$", lw=1)
axs[0, 1].plot(year, mu2[:, 1], "b-.", label=r"$p(\theta_2)$", lw=1)
axs[0, 1].plot(year, mu3[:, 1], "y-", label=r"$p(\theta_3)$", lw=1)
axs[0, 1].tick_params(axis='both', labelsize=12)
axs[0, 1].legend(fontsize='small', ncol=2, bbox_to_anchor=(
    0.45, -0.1, 0.56, 1.48), frameon=True, title='Lynx', title_fontsize='small')
axs[0, 1].text(1832, 250, s='\\textbf{b}', fontsize=20)

axs[1, 0].plot(year, data[:, 0], "ko", mfc="none",
               ms=2.5, label="observed hare")
axs[1, 0].plot(year, data[:, 1], "bx", ms=2.5, label="observed lynx")
axs[1, 0].plot(year, mu0[:, 0], "k-", lw=1.5, label="predicted hare")
axs[1, 0].plot(year, mu0[:, 1], "b-", label="predicted lynx", lw=1.5)
axs[1, 0].tick_params(axis='both', labelsize=12)
axs[1, 0].fill_between(year, pi0[0][:, 0], pi0[1][:, 0],
                       color="k", alpha=0.3, interpolate=False)
axs[1, 0].fill_between(year, pi0[0][:, 1], pi0[1][:, 1],
                       color="b", alpha=0.3, interpolate=True)
axs[1, 0].set_ylabel('Pelts (thousands)', fontsize=12)
axs[1, 0].legend(fontsize='small', ncol=2, bbox_to_anchor=(
    0.45, -0.1, 0.56, 1.1), frameon=True, title=r'$p(\theta_0)$', title_fontsize='small')
axs[1, 0].text(1825, 220, s='\\textbf{c}', fontsize=20)
axs[1, 1].plot(year, data[:, 0], "ko", mfc="none",
               ms=2.5, label="observed hare")
axs[1, 1].plot(year, data[:, 1], "bx", ms=2.5, label="observed lynx")
axs[1, 1].plot(year, mu1[:, 0], "k-", lw=1.5, label="predicted hare")
axs[1, 1].plot(year, mu1[:, 1], "b-", label="predicted lynx", lw=1.5)
axs[1, 1].fill_between(year, pi1[0][:, 0], pi1[1][:, 0],
                       color="k", alpha=0.3, interpolate=False)
axs[1, 1].fill_between(year, pi1[0][:, 1], pi1[1][:, 1],
                       color="b", alpha=0.3, interpolate=True)
axs[1, 1].text(1832, 210, s='\\textbf{d}', fontsize=20)
axs[1, 1].legend(fontsize='small', ncol=2, bbox_to_anchor=(
    0.45, -0.1, 0.56, 1.1), title=r'$p(\theta_1)$', title_fontsize='small', frameon=True)
axs[2, 0].plot(year, data[:, 0], "ko", mfc="none",
               ms=2.5, label="observed hare")
axs[2, 0].plot(year, data[:, 1], "bx", ms=2.5, label="observed lynx")
axs[2, 0].plot(year, mu2[:, 0], "k-", lw=1.5, label="predicted hare")
axs[2, 0].plot(year, mu2[:, 1], "b-", label="predicted lynx", lw=1.5)
axs[2, 0].tick_params(axis='both', labelsize=12)
axs[2, 0].fill_between(year, pi2[0][:, 0], pi2[1][:, 0],
                       color="k", alpha=0.3, interpolate=False)
axs[2, 0].fill_between(year, pi2[0][:, 1], pi2[1][:, 1],
                       color="b", alpha=0.3, interpolate=True)
axs[2, 0].text(1825, 218, s='\\textbf{e}', fontsize=20)
axs[2, 0].set_ylabel('Pelts (thousands)', fontsize=12)
axs[2, 0].legend(fontsize='small', ncol=2, bbox_to_anchor=(
    0.40, -0.1, 0.56, 1.14), title=r'$p(\theta_2)$', title_fontsize='small', frameon=True)
axs[2, 0].set_xlabel('Year', fontsize=12)
axs[2, 1].plot(year, data[:, 0], "ko", mfc="none",
               ms=2.5, label="observed hare")
axs[2, 1].plot(year, data[:, 1], "bx", ms=2.5, label="observed lynx")
axs[2, 1].plot(year, mu3[:, 0], "k-", lw=1.5, label="predicted hare")
axs[2, 1].plot(year, mu3[:, 1], "b-", label="predicted lynx", lw=1.5)
axs[2, 1].tick_params(axis='both', labelsize=12)
axs[2, 1].fill_between(year, pi3[0][:, 0], pi3[1][:, 0],
                       color="k", alpha=0.3, interpolate=False)
axs[2, 1].fill_between(year, pi3[0][:, 1], pi3[1][:, 1],
                       color="b", alpha=0.3, interpolate=True)
axs[2, 1].text(1832, 208, s='\\textbf{f}', fontsize=20)
axs[2, 1].legend(fontsize='small', ncol=1, bbox_to_anchor=(
    0.8, -0.1, 0.56, 1.14), title=r'$p(\theta_3)$', title_fontsize='small', frameon=True)
axs[2, 1].set_xlabel('Year', fontsize=12)
fig.subplots_adjust(wspace=0.12, hspace=0.05)
fig.set_figwidth(6.5)
fig.set_figheight(6.5)
plt.savefig("./odesppcall.pdf", dpi=1000)
%
