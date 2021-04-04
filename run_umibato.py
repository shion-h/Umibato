#
# run_umibato.py
#
# Copyright (c) 2020 Shion Hosoda
#
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php
#

import argparse
import pandas as pd
import importlib
from umibato import Umibato


parser = argparse.ArgumentParser()
parser.add_argument('qmps_filepath', help='Quantitative microbiota profiles file path')
parser.add_argument('metadata_filepath', 
                    help='Metadata file (including \"subjectID\" and \"timepoint\" columns) path')
parser.add_argument('k_min', help='The min number of states', type=int)
parser.add_argument('k_max', help='The max number of states', type=int)
parser.add_argument('-s', '--k_step', help='the step number of states',
                    type=int, default=1)
parser.add_argument('-g', '--gp_correction', help='Use of GP correction.', 
                    default=False, action='store_true')
parser.add_argument('--no_gp_correction', help='No use of GP correction.', 
                    dest='gp_correction', action='store_false')
parser.add_argument('-a', '--augmentation_size', help='The number of data augmented by Gaussian process regression.',
                    type=int, default=0)
parser.add_argument('-r', '--x_standardization', help='Standardize x.', 
                    default=True, action='store_true')
parser.add_argument('--no_x_standardization', help='Not standardize x.', 
                    dest='x_standardization', action='store_false')
parser.add_argument('-l', '--y_var_lower_bound', help='The lower bound of y variances.', 
                    type=float, default=1.0e-4)
parser.add_argument('-v', '--est_y_var', help='Use the different variances of y for each observation point (estimated by GPR).', 
                    default=True, action='store_true')
parser.add_argument('--no_est_y_var', help='Not use the different variances of y.', 
                    dest='est_y_var', action='store_false')
parser.add_argument('-m', '--max_iter', help='The max number of iterations', 
                    type=int, default=100)
parser.add_argument('-c', '--tol', help='Convergence threshold',
                    type=float, default=1.0e-4)
parser.add_argument('-i', '--n_init', help='The number of trials', 
                    type=int, default=10)
parser.add_argument('-j', '--n_jobs', help='The number of jobs', 
                    type=int, default=1)
parser.add_argument('-t', '--ra_threshold', help='Threshold of the relative abundance',
                    type=float, default=0.0)
parser.add_argument('-o', '--output_path', help='Path of an output directory',
                    type=str, default='.')
args = parser.parse_args()
qmps = pd.read_csv(args.qmps_filepath, delimiter='\t', index_col=0)
qmps.columns = qmps.columns.astype(str)
normalized_qmps = qmps.apply(lambda x: x/x.sum(), axis=0)
qmps = qmps.loc[normalized_qmps.mean(1)>args.ra_threshold, :]
metadata = pd.read_csv(args.metadata_filepath, delimiter='\t', index_col=0)
metadata.index = metadata.index.astype(str)
obj = Umibato(k_min=args.k_min, k_max=args.k_max, k_step=args.k_step,
              n_init=args.n_init, n_jobs=args.n_jobs,
              gp_correction=args.gp_correction,
              augmentation_size=args.augmentation_size,
              x_standardization=args.x_standardization,
              y_var_lower_bound=args.y_var_lower_bound,
              est_y_var=args.est_y_var,
              max_iter=args.max_iter, tol=args.tol,
              output_path=args.output_path)
obj.fit(qmps, metadata)
obj.plot()
