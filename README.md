# Umibato
## Description
Umibato is a method for estimating time-varying microbial interactions using time-series quantitative data based on the Lotka-Volterra equation.
Details for this method are described in [this paper](https://www.biorxiv.org/content/10.1101/2021.01.28.428580v2).
Umibato is implemented as a python Class.
Variational inference for CTRHMM, the second step of Umibato, is written in C++.
You can also perform Umibato on UNIX shells using run_umibato.py.
## Directories
- build
    - an empty directory for cmake compilation
- data
    - contains toy data for Umibato tutorials
- notebook
    - contains a notebook for make testcases
- src
    - C++ source code for estimation of CTRHMM parameters
- testcase
    - testcase data for CTRHMM implementation
- tutorial
    - contains a notebook for the tutorials
- umibato
    - the Python library directory
## Requirements
- Python (3.7.7)
- Python library
    - numpy (1.18.1)
    - pandas (1.0.3)
    - GPy (1.9.9)
    - matplotlib (3.1.3)
    - seaborn (0.10.1)
- g++ Compiler (5.4.0) or clang (12.0.0)
- C++ library
    - boost (1.63.0)
    - eigen (included in this repository)
- cmake
## Compilation
- After preparing the requirements, perform the following:
```
cmake -B build
cmake --build build
```
- Then, you can use Umibato from the root directry.
## Usage 
python run_umibato.py
[-h] [-s K_STEP] [-g] [--no_gp_correction]
[-a AUGMENTATION_SIZE] [-r] [--no_x_standardization]
[-l Y_VAR_LOWER_BOUND] [-v] [--no_est_y_var]
[-m MAX_ITER] [-c TOL] [-i N_INIT] [-j N_JOBS]
[-t RA_THRESHOLD] [-o OUTPUT_PATH]
qmps_filepath metadata_filepath k_min k_max
### Positional arguments
- **qmps_filepath**
    - Quantitative microbiota profiles file path.
- **metadata_filepath**
    - Metadata file (including "subjectID" and "timepoint" columns) path.
- **k_min**
    - The min number of states.
- **k_max**
    - The max number of states.

### Optional arguments
- -h, --help
    - Show the help message and exit.
- -s K_STEP, --k_step K_STEP
    - The step number of states (default: 1).
- -g, --gp_correction
    - Use of GP correction (default: False).
    - --no_gp_correction
        - No use of GP correction.
- -a AUGMENTATION_SIZE, --augmentation_size AUGMENTATION_SIZE
    - The number of data augmented by Gaussian process regression (default: 0).
- -r, --x_standardization
    - Standardize x (default: True).
    - --no_x_standardization
        - Not standardize x.
- -l Y_VAR_LOWER_BOUND, --y_var_lower_bound Y_VAR_LOWER_BOUND
    - The lower bound of y variances (default: 1.0e-4) when using estimation of y variances. 
- -v, --est_y_var
    - Use estimation of y variances by GPR for each observation point (default: True).
    - --no_est_y_var
        - Not use estimation of y variances.
- -m MAX_ITER, --max_iter MAX_ITER
    - The max number of iterations (default: 100).
- -c TOL, --tol TOL
    - Convergence threshold (default: 1.0e-4).
- -i N_INIT, --n_init N_INIT
    - The number of trials (default: 1.0e-4).
- -j N_JOBS, --n_jobs N_JOBS
    - The number of jobs (default: 1).
- -t RA_THRESHOLD, --ra_threshold RA_THRESHOLD
    - Threshold of relative abundance (default: 0.0).
- -o OUTPUT_PATH, --output_path OUTPUT_PATH
    - Path to an output directory (default: '.').
