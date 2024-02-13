# BayesianODE-PriorImpactAssessment
This repository has code for the paper Prior impact assessment for dynamical systems described by ordinary differential equations.
## examples
The examples folder contains two subfolders: lotka_volterra and the SEIR.
### lotka_volterra
The last two scripts to be executed should be  'wasser_dist_prior.py'  and  'wasser_exlot.py' in that order, as they use results from other scripts.
### SEIR
This folder contains a subfolder called prior_samples and additional scripts. All of the scripts and the subfolder should be executed before running 'wasser_results.py' and 'wd_mar_ex.py'. 
#### Prior_samples
The subfolder contains scripts for sampling from prior distributions. Running these scripts on the HPC is preferable as they take longer. OR use the script `sample_batches.py` for batch sampling to reduce execution time.


