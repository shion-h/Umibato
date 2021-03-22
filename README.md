# Umibato
## Description
Umibato is a method for estimating time-varying microbial interactions using time-series quantitative data based on the Lotka-Volterra equation.
Details for this method are described in [this paper](https://www.biorxiv.org/content/10.1101/2021.01.28.428580v2).
Umibato is implemented as a python Class.
Variational inference for CTRHMM, the second step of Umibato, is written in C++.
You can also perform Umibato on UNIX shells using run_umibato.py.
## Tutorials
### Docker tutorial
Here, Umibato tutorial using Docker is provided.
We tested this tutorial using Docker desktop version 3.2.2 on MacOS version 11.2.3. 
After cloning this repository, perform the following command on the root directory to build a docker image:
```
docker build -t umibato:1 .
```
"umibato:1" is an arbitary name. Now, all installations and compilations were finished. Next, you constract and run a container using the following command:
```
docker container run -it --name umibato-1 umibato:1 /bin/bash
```
"umibato-1" is also an arbitary name. You've been logged in the virtual Docker machine. 
**Now, you can use Umibato.** 

The following shows the commands to reproduce the results.

Let's download the dataset used in the paper. Perform the following commands in the docker shell:
```
mkdir -p ./data/bucci_et_al
curl -o ./data/bucci_et_al/metadata.tsv https://bitbucket.org/MDSINE/mdsine/raw/a5384a34f4c75402aee9bdb8b90db3d70052ac73/data_diet/metadata.txt
curl -o ./data/bucci_et_al/x.tsv https://bitbucket.org/MDSINE/mdsine/raw/a5384a34f4c75402aee9bdb8b90db3d70052ac73/data_diet/counts.txt
```
This dataset is from the following paper:

> Bucci, Vanni, et al. "MDSINE: Microbial Dynamical Systems INference Engine for microbiome time-series analyses." Genome biology 17.1 (2016): 1-17.

Rename a column name of the original dataset.
```
sed -e 's/measurementID/timepoint/g' ./data/bucci_et_al/metadata.tsv > ./tmp.tsv
cp ./tmp.tsv ./data/bucci_et_al/metadata.tsv
rm ./tmp.tsv
```
The following command performs Umibato with 10 CTRHMM trials and one-core CPU:
```
python3 run_umibato.py ./data/bucci_et_al/x.tsv ./data/bucci_et_al/metadata.tsv 1 15 --n_init 10 --n_jobs 1 --output_path ./output
```
And you can see some figures in /home/output/figures. 
The following command performs the same experiment as the paper:
```
python3 run_umibato.py ./data/bucci_et_al/x.tsv ./data/bucci_et_al/metadata.tsv 1 15 --n_init 10000 --n_jobs 100 --output_path ./output
```
This command takes 2-3 hours for 100 parallel. It takes 1 day or more to perform Umibato for 2 parallel.
### Notebook tutorial
To use the tutorial notebook, you need requirements and compilations below or using Docker. 
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
- Libraries
    - Python (3.7.7)
    - Python library
        - tqdm (4.48.2)
        - numpy (1.18.1)
        - pandas (1.0.3)
        - GPy (1.9.9)
        - matplotlib (3.1.3)
        - seaborn (0.10.1)
    - g++ Compiler (5.4.0) or clang (12.0.0)
    - C++ library
        - boost (1.63.0)
        - eigen (included in this repository)
    - cmake (3.16.3)
- Dataset
    - Quantitative abundance profiles (QMPs)
        - The rows and columns indicate microbes and samples, respectively.
    - Metadata
        - The rows and columns indicate samples and factors of metadata, respectively.
        - including "timepoint" and "subjectID" columns (see an example of data/toy/metadata.tsv).
## Compilation
- After preparing the requirements, perform the following:
```
cmake -B build
cmake --build build
```
- Then, you can use Umibato from the root directry.
## Output
### "results" directory
The results of all trials are here.
### "best_results" directory
- ELBO.csv
    - ELBO value at each iteration.
- Q.csv
    - Transition rate matrix. Rows and columns indicate source and destination microbes, respectively.
- ViterbiPath.csv
    - Maximum likelihood path of interaction states.
- phi{state number}.csv
    - gLVE parameters estimated by CTRHMM.
- interaction_parameters{state number}.csv
    - Corrected gLVE parameters (phi{state number}.csv) if using standardizing X.
    - This file would not be output if not using standardizing X.
### "figures" directory
- gp_regression.pdf
    - Shows the results of Gaussian process regression.
- interaction_networks.pdf
    - Shows the relative interaction intensities (= the interaction parameters divided by the standard deviation of growth rate across series) using Umibato (same as figure 5 in the paper).
- max_maximized_elbo.pdf
    - Shows the results of ELBOs (same as figure S5 in the paper).
- maximized_elbo_boxplot.pdf
    - Shows the distribution of ELBO values of all trials.
- viterbi_path.pdf
    - Shows maximum likelihood path of interaction states for each subject (same as figure 3 in the paper).
### "processed_data" directory
- metadata.csv
    - Preprocessed metadata (including subject and timepoint information).
- x.csv
    - Preprocessed QMPs (preprocess is standardization, for example).
- y.csv
    - Estimated growth rate expectations.
- yVariance.csv
    - Estimated growth rate variances.
### KinitTrialELBO.csv
- The summary table of ELBO values for each K_init for each trial.
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
        - GP correction replace abundances (variable X) into exponentials of prediction values of GPR (e^E[f]).
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
