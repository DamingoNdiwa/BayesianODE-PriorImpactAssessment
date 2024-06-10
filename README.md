# Wasserstein distance prior impact assessment for ODE models

## Introduction

This repository contains supporting code for the pre-print:

Mingo, D. N., Hale, J. S. and Ley, C.,: Bayesian prior impact assessment for
dynamical systems described by ordinary differential equations.

**TODO: Add link to preprint**

The code is archived at:

**TODO: Add link to Zenodo repository**

The code in this repository is licensed under the GNU Lesser General Public
License version 3 or later, see `COPYING` and `COPYING.LESSER`.

## Examples

### Lotka-Volterra

Scripts to reproduce the results for the Lotka-Volterra example are contained
in `examples/lotka_volterra`.

In order, run:
1. `wasser_exlot.py`
2. `wasser_dist_prior.py`
3. `wasser_dist_ex.py`

The scripts `lotka_priors_ppc.py` and `pairplot.py` can be executed
independently.

### SEIR

Scripts to reproduce the results for the SEIR example are contained
in `examples/SEIR`.

In order, run:
1. All files in `prior_samples/`.
2. `run_seirpost.sh` executes `wasser_seir.py` and saves the samples.
3. `wasser_seir.py`  
4. `wd_mar_ex.py`

#### Prior samples (SEIR)

The subfolder `examples/SEIR/prior_samples` contains scripts for sampling from
prior distributions. Running these scripts on an HPC is preferable as they take
longer, or use the script `sample_batches.py` for batch sampling to reduce
execution time.

### Additional scripts

The script `example_diagnostics.py` demonstrates how to perform Geweke
diagnostics to check for model convergence. The script requires uploading
posterior samples and specifying the model (SEIR or Lotka-Volterra). 

[![DOI](https://zenodo.org/badge/605141673.svg)](https://zenodo.org/doi/10.5281/zenodo.10202092)
