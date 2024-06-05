# BayesianODE-PriorImpactAssessment
This repository has code for the paper Prior impact assessment for dynamical systems described by ordinary differential equations.
## examples
The examples folder contains two subfolders: lotka_volterra and the SEIR.
### lotka_volterra
The scripts `utils_lotka.py` and `utils_summary.py` contain helper functions.<br>
The scripts:<br>
1. `wasser_exlot.py`<br>
2. `wasser_dist_prior.py`<br>
3. `wasser_dist_ex.py`<br>
should be executed in that order.<br>
The scripts `lotka_priors_ppc.py` and `pairplot.py` can be executed at any time as they are standalone.

### SEIR
This folder contains a subfolder called prior_samples and additional scripts. All of the scripts and the subfolder should be executed before running `wasser_results.py` and `wd_mar_ex.py`. 
#### Prior_samples
The subfolder contains scripts for sampling from prior distributions. Running these scripts on the HPC is preferable as they take longer. OR use the script `sample_batches.py` for batch sampling to reduce execution time.

## Scripts
The script `example_diagnostics.py` demonstrates how to perform Geweke diagnostics to check for model convergence. 
The script requires uploading posterior samples and specifying the model (SEIR or Lotka-Volterra). Also, `utils_diagnostics.py` has a  function used by `example_diagnostics.py`.
