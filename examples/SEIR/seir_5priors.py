import jax
import pandas as pd
import matplotlib.pyplot as plt
from jax import random
from jax.random import PRNGKey
import jax.numpy as jnp
from jax.config import config
import matplotlib.pyplot as plt
import scienceplots
from matplotlib.lines import Line2D
from utils_seir import joint_pos, post_samples, post_samples1

config.update("jax_enable_x64", True)
plt.style.use(['science', 'ieee'])


# load data
data_plot = pd.read_csv("filtered_data.csv")

num_results = 3000
num_burnin_steps = 1000
key = 1

# First prior 

pos_samples0 = post_samples(num_results, num_burnin_steps,
                            key, phi=5)
# Second prior
pos_samples1 = post_samples(num_results, num_burnin_steps,
                            key, phi=42) 
# Third prior

pos_samples2 = post_samples(num_results, num_burnin_steps,
                            key, phi=1)
# fourth prior
pos_samples3 = post_samples1(num_results, num_burnin_steps,
                             key, a=16)
# fifth prior
pos_samples4 = post_samples(num_results, num_burnin_steps,
                            key, phi=150)
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

# For the fourth prior

posterior_predictive3 = jax.vmap(
    lambda key,
    sample: dist.sample(
        seed=key,
        value=sample))(
            keys,
    pos_samples3)

mu3 = jnp.mean(posterior_predictive3.y, axis=0)
pi3 = jnp.percentile(posterior_predictive3.y, jnp.array([25, 75]), axis=0)

# For the fifth prior

posterior_predictive4 = jax.vmap(
    lambda key,
    sample: dist.sample(
        seed=key,
        value=sample))(
            keys,
    pos_samples4)

mu4 = jnp.mean(posterior_predictive4.y, axis=0)
pi4 = jnp.percentile(posterior_predictive4.y, jnp.array([25, 75]), axis=0)

# Get the plots
widths = [500, 500]
heights = [1000, 1000, 1000]
fig = plt.figure(constrained_layout=False)
gs = fig.add_gridspec(
    3,
    2,
    width_ratios=widths,
    height_ratios=heights,
    wspace=0.2,
    hspace=0.15)

# Define the top subplot spanning both columns
f3_ax1 = fig.add_subplot(gs[0, 0])
f3_ax1.plot(mu0, "k-", label=r"$p_0$")
f3_ax1.plot(mu1, "b-", label=r"$p_1$")
f3_ax1.plot(mu2, "r-", label=r"$p_2$", alpha=0.5)
f3_ax1.plot(mu3, "c-", label=r"$p_3$")
f3_ax1.plot(mu4, "y-", label=r"$p_4$")

f3_ax1.set_xticks([0, 30, 60, 90, 120])
f3_ax1.set_xticklabels(['29-Feb',
                        '31-March',
                        '30-April',
                        '30-May',
                        '29-June'],
                       rotation=12)
f3_ax1.legend(fontsize='small', ncol=1, frameon=True)
f3_ax1.set_ylim(0, 310)  # Set the maximum y-limit to 250
f3_ax1.set_xticks([])
f3_ax1.set_xticklabels([])
# Define the four subplots forming a 2x2 grid below the top subplot
f3_ax22 = fig.add_subplot(gs[2, 1], sharey=f3_ax1)
f3_ax22.plot(
    data_plot.new_cases.values,
    "ko",
    mfc="none",
    ms=1.5,
    label="observed",
    alpha=0.5)
f3_ax22.plot(
    mu4,
    "y-",
    label=r"$\phi^{-1} \stackrel{\text{iid}}{\sim} \text{Exponential}(150)$")
f3_ax22.fill_between(data_plot.date.values[:-1],
                     pi4[0],
                     pi4[1],
                     color="blue",
                     alpha=0.4,
                     interpolate=False,
                     label=r'$50\%$ pointwise CIs')
f3_ax22.set_xticks([0, 30, 60, 90, 120])
f3_ax22.set_xticklabels(['29-Feb',
                         '31-March',
                         '30-April',
                         '30-May',
                         '29-June'],
                        rotation=20)
legend_elements = [
    Line2D(
        [0],
        [0],
        color='y',
        ls='-',
        label=r"$\phi^{-1} \stackrel{\text{iid}}{\sim} \text{Exponential}(150)$")]
f3_ax22.legend(
    handles=legend_elements,
    loc='best',
    fontsize='x-small',
    frameon=True,
    title=r'$p_4$',
    title_fontsize='small')
f3_ax22.set_ylim(0, 310)  # Set the maximum y-limit to 250
f3_ax2 = fig.add_subplot(gs[0, 1], sharey=f3_ax1)
f3_ax2.plot(
    data_plot.new_cases.values,
    "ko",
    mfc="none",
    ms=1.5,
    label="observed",
    alpha=0.5)
f3_ax2.plot(
    mu0,
    "k-",
    label=r"$\phi^{-1} \stackrel{\text{iid}}{\sim} \text{Exponential}(5)$")
f3_ax2.fill_between(data_plot.date.values[:-1],
                    pi0[0],
                    pi0[1],
                    color="blue",
                    alpha=0.4,
                    interpolate=False,
                    label=r'$50\%$ pointwise CIs')
f3_ax2.set_xticks([])
f3_ax2.set_xticklabels([])
legend_elements = [
    Line2D(
        [0],
        [0],
        color='k',
        ls='-',
        label=r"$\phi^{-1} \stackrel{\text{iid}}{\sim} \text{Exponential}(5)$")]
f3_ax2.legend(
    handles=legend_elements,
    loc='best',
    fontsize='small',
    frameon=True,
    title=r'$p_0$',
    title_fontsize='small')
f3_ax2.set_ylim(0, 310)  # Set the maximum y-limit to 250

f3_ax3 = fig.add_subplot(gs[1, 0], sharey=f3_ax1)
f3_ax3.plot(
    data_plot.new_cases.values,
    "ko",
    mfc="none",
    ms=1.5,
    label="observed",
    alpha=0.5)
f3_ax3.plot(
    mu1,
    "b-",
    label=r"$\phi^{-1} \stackrel{\text{iid}}{\sim} \text{Exponential}(42)$")
f3_ax3.fill_between(data_plot.date.values[:-1],
                    pi1[0],
                    pi1[1],
                    color="blue",
                    alpha=0.4,
                    interpolate=False,
                    label=r'$50\%$ pointwise CIs')
f3_ax3.set_xticks([])
f3_ax3.set_xticklabels([])
legend_elements = [
    Line2D(
        [0],
        [0],
        color='b',
        ls='-',
        label=r"$\phi^{-1} \stackrel{\text{iid}}{\sim} \text{Exponential}(42)$")]
f3_ax3.legend(
    handles=legend_elements,
    loc='best',
    fontsize='small',
    frameon=True,
    title=r'$p_1$',
    title_fontsize='small')
f3_ax3.set_ylim(0, 310)  # Set the maximum y-limit to 250

f3_ax4 = fig.add_subplot(gs[1, 1], sharey=f3_ax1)
f3_ax4.plot(
    mu2,
    "r-",
    label=r"$\phi^{-1} \stackrel{\text{iid}}{\sim} \text{Exponential}(1)$")
f3_ax4.fill_between(data_plot.date.values[:-1],
                    pi2[0],
                    pi2[1],
                    color="blue",
                    alpha=0.4,
                    interpolate=False,
                    label=r'$50\%$ pointwise CIs')
f3_ax4.plot(
    data_plot.new_cases.values,
    "ko",
    mfc="none",
    ms=1.5,
    label="observed",
    alpha=0.5)
f3_ax4.set_xticks([])
f3_ax4.set_xticklabels([])
legend_elements = [
    Line2D(
        [0],
        [0],
        color='r',
        ls='-',
        label=r"$\phi^{-1} \stackrel{\text{iid}}{\sim} \text{Exponential}(1)$")]
f3_ax4.legend(
    handles=legend_elements,
    loc='best',
    fontsize='small',
    frameon=True,
    title=r'$p_2$',
    title_fontsize='small')

f3_ax4.set_ylim(0, 350)  # Set the maximum y-limit to 250

f3_ax5 = fig.add_subplot(gs[2, 0], sharey=f3_ax4)
f3_ax5.plot(
    data_plot.new_cases.values,
    "ko",
    mfc="none",
    ms=1.5,
    label="observed",
    alpha=0.5)
f3_ax5.plot(mu3, "c-", label=r"$p_2$", alpha=0.5)
f3_ax5.fill_between(data_plot.date.values[:-1],
                    pi3[0],
                    pi3[1],
                    color="blue",
                    alpha=0.45,
                    interpolate=False,
                    label=r"$\phi^{-1} \stackrel{\text{iid}}{\sim} \text{Gamma}(16, 16)$")
f3_ax5.set_xticks([0, 30, 60, 90, 120])
f3_ax5.set_xticklabels(['29-Feb',
                        '31-March',
                        '30-April',
                        '30-May',
                        '29-June'],
                       rotation=20)
f3_ax5.set_ylabel('Number of daily new cases', fontsize=10, y=1.5)
legend_elements = [
    Line2D(
        [0],
        [0],
        color='c',
        ls='-',
        label=r"$\phi^{-1} \stackrel{\text{iid}}{\sim} \text{Gamma}(16, 16)$")]
f3_ax5.legend(
    handles=legend_elements,
    loc='best',
    fontsize='small',
    frameon=True,
    title=r'$p_3$',
    title_fontsize='small')
f3_ax5.set_xlabel('Date', fontsize=10, x=1.1)
fig.set_figwidth(6.2)
fig.set_figheight(3.2)
fig.savefig("./ppcseirlux.pdf", dpi=1000)

# Another approach for the plots

fig, axs = plt.subplots(3, 2, sharex=True, sharey=True)
axs[0, 0].text(-20, 350, r"$\textbf{a}$", fontsize='14')
axs[0, 0].set_ylim(0, 350)  # Set the maximum y-limit to 250
axs[0, 0].plot(mu0, "k-", label=r"$p_0$")
axs[0, 0].plot(mu1, "b-", label=r"$p_1$")
axs[0, 0].plot(mu2, "r-", label=r"$p_2$", alpha=0.5)
axs[0, 0].plot(mu3, "c-", label=r"$p_3$")
axs[0, 0].plot(mu4, "y-", label=r"$p_4$")
axs[0, 0].set_xticks([0, 30, 60, 90, 120])
axs[0, 0].set_xticklabels(['29-Feb',
                           '31-March',
                           '30-April',
                           '30-May',
                           '29-June'],
                          rotation=12)
axs[0, 0].legend(fontsize='x-small', ncol=2, frameon=True)
axs[0, 1].text(-12, 350, r"$\textbf{b}$", fontsize='14')
axs[0, 1].set_ylim(0, 350)
axs[0, 1].plot(data_plot.new_cases.values, "ko", mfc="none",
               ms=1.5, label="observed", alpha=0.5)
axs[0, 1].plot(
    mu0, "k-", label=r"$\phi^{-1} \stackrel{\text{iid}}{\sim} \text{Exponential}(5)$")
axs[0,
    1].fill_between(data_plot.date.values[:-1],
                    pi0[0],
                    pi0[1],
                    color="blue",
                    alpha=0.4,
                    interpolate=False,
                    label=r'$50\%$ pointwise CIs')
axs[0, 1].tick_params(axis='both', labelsize=10)
legend_elements = [
    Line2D(
        [0],
        [0],
        color='k',
        ls='-',
        label=r"$\phi^{-1} \stackrel{\text{iid}}{\sim} \text{Exponential}(5)$")]
axs[1, 0].set_ylim(0, 350)
axs[0,
    1].legend(handles=legend_elements,
              loc='best',
              fontsize='x-small',
              frameon=True,
              title=r'$p_0$',
              title_fontsize='x-small')
axs[1, 0].plot(data_plot.new_cases.values, "ko", mfc="none",
               ms=1.5, label="observed", alpha=0.5)
axs[1, 0].plot(
    mu1, "b-", label=r"$\phi^{-1} \stackrel{\text{iid}}{\sim} \text{Exponential}(42)$")
axs[1,
    0].fill_between(data_plot.date.values[:-1],
                    pi1[0],
                    pi1[1],
                    color="blue",
                    alpha=0.4,
                    interpolate=False,
                    label=r'$50\%$ pointwise CIs')
legend_elements = [
    Line2D(
        [0],
        [0],
        color='b',
        ls='-',
        label=r"$\phi^{-1} \stackrel{\text{iid}}{\sim} \text{Exponential}(42)$")]
axs[1,
    0].legend(handles=legend_elements,
              loc='best',
              fontsize='x-small',
              frameon=True,
              title=r'$p_1$',
              title_fontsize='x-small')
axs[1, 0].plot(data_plot.new_cases.values, "ko", mfc="none",
               ms=1.5, label="observed", alpha=0.5)
axs[1, 0].text(-20, 350, r"$\textbf{c}$", fontsize='14')
axs[1, 1].text(-12, 350, r"$\textbf{d}$", fontsize='14')
axs[1, 1].set_ylim(0, 350)
axs[1, 1].plot(
    mu2, "r-", label=r"$\phi^{-1} \stackrel{\text{iid}}{\sim} \text{Exponential}(1)$")
axs[1,
    1].fill_between(data_plot.date.values[:-1],
                    pi2[0],
                    pi2[1],
                    color="blue",
                    alpha=0.4,
                    interpolate=False,
                    label=r'$50\%$ pointwise CIs')
axs[1, 1].plot(data_plot.new_cases.values, "ko", mfc="none",
               ms=1.5, label="observed", alpha=0.5)
axs[1, 1].set_xticks([0, 30, 60, 90, 120])
axs[1, 1].set_xticklabels(['29-Feb',
                           '31-March',
                           '30-April',
                           '30-May',
                           '29-June'],
                          rotation=28)
legend_elements = [
    Line2D(
        [0],
        [0],
        color='r',
        ls='-',
        label=r"$\phi^{-1} \stackrel{\text{iid}}{\sim} \text{Exponential}(1)$")]
axs[1,
    1].legend(handles=legend_elements,
              loc='best',
              fontsize='x-small',
              frameon=True,
              title=r'$p_2$',
              title_fontsize='x-small')
axs[2, 0].text(-15, 350, r"$\textbf{e}$", fontsize='14')
axs[2, 0].set_ylim(0, 350)
axs[2, 0].plot(data_plot.new_cases.values, "ko", mfc="none",
               ms=1.5, label="observed", alpha=0.5)
axs[2, 0].plot(mu2, "c-", label=r"$p(\theta_2)$", alpha=0.5)
axs[2,
    0].fill_between(data_plot.date.values[:-1],
                    pi3[0],
                    pi3[1],
                    color="blue",
                    alpha=0.45,
                    interpolate=False,
                    label=r"$\phi^{-1} \stackrel{\text{iid}}{\sim} \text{Gamma}(16, 16)$")
axs[2, 0].set_xticks([0, 30, 60, 90, 120])
axs[2, 0].set_xticklabels(['29-Feb',
                           '31-March',
                           '30-April',
                           '30-May',
                           '29-June'],
                          rotation=28)
axs[2, 0].set_ylabel('Number of daily new cases', fontsize=10, y=1.5)
legend_elements = [
    Line2D(
        [0],
        [0],
        color='c',
        ls='-',
        label=r"$\phi^{-1} \stackrel{\text{iid}}{\sim} \text{Gamma}(16, 16)$")]
axs[2,
    0].legend(handles=legend_elements,
              loc='best',
              fontsize='x-small',
              frameon=True,
              title=r'$p_3$',
              title_fontsize='x-small')
axs[2, 0].set_xlabel('Date', fontsize=10, x=1.1)
axs[2, 1].text(-12, 350, r"$\textbf{f}$", fontsize='14')
axs[2, 1].set_ylim(0, 350)
axs[2, 1].plot(data_plot.new_cases.values, "ko", mfc="none",
               ms=1.5, label="observed", alpha=0.5)
axs[2, 1].plot(
    mu4, "y-", label=r"$\phi^{-1} \stackrel{\text{iid}}{\sim} \text{Exponential}(150)$")
axs[2,
    1].fill_between(data_plot.date.values[:-1],
                    pi4[0],
                    pi4[1],
                    color="blue",
                    alpha=0.4,
                    interpolate=False,
                    label=r'$50\%$ pointwise CIs')
axs[2, 1].set_xticks([0, 30, 60, 90, 120])
axs[2, 1].set_xticklabels(['29-Feb',
                           '31-March',
                           '30-April',
                           '30-May',
                           '29-June'],
                          rotation=20)
legend_elements = [
    Line2D(
        [0],
        [0],
        color='y',
        ls='-',
        label=r"$\phi^{-1} \stackrel{\text{iid}}{\sim} \text{Exponential}(150)$")]
axs[2,
    1].legend(handles=legend_elements,
              loc='best',
              fontsize='x-small',
              frameon=True,
              title=r'$p_4$',
              title_fontsize='small')
fig.set_figwidth(5.5)
fig.set_figheight(5)
# Increase the spacing between subplots
plt.subplots_adjust(hspace=0.2, wspace=0.12)
fig.savefig("./ppcseirlux2.pdf", dpi=1000)
