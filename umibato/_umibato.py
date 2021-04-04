#
# _umibato.py
#
# Copyright (c) 2020 Shion Hosoda
#
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php
#

import os
import sys
import math
import shutil
import logging
logging.getLogger().setLevel(logging.WARNING)
import warnings
warnings.filterwarnings('ignore')
import subprocess
from multiprocessing import Pool
from tqdm.autonotebook import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
from ._gpmodule import fit_gp_model, estimate_grad_variance
from ._plotmodule import plot_state, plot_directed_network


class Umibato(object):
    def __init__(self, k_min=1, k_max=10, k_step=1, augmentation_size=0,
                 gp_correction=False, x_standardization=True,
                 y_var_lower_bound=1e-4, est_y_var=True,
                 max_iter=100, tol=1e-4,
                 n_init=100, n_jobs=5, output_path='.'):
        self.K_list = list(range(k_min, k_max+1, k_step))
        self.augmentation_size = augmentation_size
        self.gp_correction = gp_correction
        self.x_standardization = x_standardization
        self.y_var_lower_bound = y_var_lower_bound
        self.est_y_var = est_y_var
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.n_jobs = n_jobs
        if output_path[-1] == '/':
            self.output_path = output_path[:-1]
        else:
            self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)

    def fit(self, qmps, metadata):
        self._estimate_growthrates(qmps, metadata)
        self._estimate_interactions()
        self._copy_best_results()
        if self.x_standardization:
            self._modify_interaction_param()

    def _estimate_growthrates(self, qmps, metadata):
        x_list = []
        y_list = []
        y_var_list = []
        timepoint_list = []
        metadata_list = []
        self.model_srs_list = []
        if set(qmps.columns) != set(metadata.index):
            print('The QMPs column name and the metadata index name must match.')
            sys.exit(1)
        self.metadata = metadata
        pseudo_abundance = 10**(math.floor(np.log10(qmps[qmps!=0].min().min())))
        ln_qmps = np.log(qmps.replace(0, pseudo_abundance))
        bacteria_list = ln_qmps.index.tolist()
        # ln_quantitative_abundance_table_with_metadata
        ln_meta = ln_qmps.T.join(metadata)
        ln_meta = ln_meta.sort_values(by=['subjectID', 'timepoint'])
        self.subject_list = metadata['subjectID'].value_counts().index.tolist()
        print('Fitting Gaussian process regression...')
        for subject in tqdm(self.subject_list):
            model_srs, x_s, y_s, y_s_var, this_metadata = \
            self._estimate_growthrates_for_each_subject(ln_meta, 
                                                        subject, bacteria_list)
            self.model_srs_list.append(model_srs)
            x_list.append(x_s)
            y_list.append(y_s)
            y_var_list.append(y_s_var)
            metadata_list.append(this_metadata)
        self._output_xy(x_list, y_list, y_var_list, metadata_list, 
                        self.y_var_lower_bound)

    def _estimate_growthrates_for_each_subject(self, ln_meta, 
                                               subject, bacteria_list):
        this_ln_meta = ln_meta[ln_meta['subjectID']==subject]
        this_ln = this_ln_meta[bacteria_list]
        timepoint = this_ln_meta['timepoint'].astype(float)
        timepoint.index = timepoint.index.astype(str)
        if self.augmentation_size > 0:
            augmented_timepoints = \
            np.random.uniform(timepoint.min(), timepoint.max(), 
                              size=self.augmentation_size)
            timepoint = timepoint.append(pd.Series(augmented_timepoints))
            timepoint = timepoint.sort_values().reset_index()
            timepoint = timepoint.drop(['index'], axis=1).iloc[:,0]
            timepoint.index = timepoint.index.astype(str)
            timepoint.name = 'timepoint'
        model_srs = this_ln.apply(lambda x: fit_gp_model(this_ln_meta['timepoint'], x), axis=0)
        x_s = None
        this_metadata = None
        if self.gp_correction | (self.augmentation_size > 0):
            this_metadata = timepoint.to_frame()
            this_metadata['subjectID'] = subject
            x_s_predicted = model_srs.apply(lambda model: model.predict(np.array(timepoint).reshape(-1, 1)))
            x_s_expected = x_s_predicted.apply(lambda x: x[0].reshape(-1))
            x_s = pd.DataFrame(list(x_s_expected))
            x_s.index = x_s_expected.index
            x_s.columns = timepoint.index
            x_s = np.exp(x_s)
        else:
            x_s = np.exp(this_ln.T)
        y_expected = model_srs.apply(lambda model: model.predictive_gradients(timepoint[:, None])[0].reshape(-1))
        y_s = pd.DataFrame(list(y_expected), index=y_expected.index, 
                           columns = timepoint.index)
        y_s_var = None
        if self.est_y_var:
            y_s_var = model_srs.apply(lambda model: estimate_grad_variance(model, timepoint[:, None]))
            y_s_var = pd.DataFrame(list(y_s_var), index=y_s_var.index, 
                                   columns=timepoint.index)
        else:
            y_s_var = y_s.copy()
            y_s_var.loc[:, :] = 1
        return model_srs, x_s, y_s, y_s_var, this_metadata

    def _output_xy(self, x_list, y_list, y_var_list, metadata_list, 
                   y_var_lower_bound):
        x = pd.concat(x_list, axis=1)
        y = pd.concat(y_list, axis=1)
        y_var = pd.concat(y_var_list, axis=1)
        if self.augmentation_size > 0:
            self.metadata = pd.concat(metadata_list, axis=0).reset_index()
            x.columns = self.metadata.index
            y.columns = self.metadata.index
            y_var.columns = self.metadata.index
        y_var = y_var[y_var > y_var_lower_bound].fillna(y_var_lower_bound)
        x = x.loc[:, self.metadata.index]
        y = y.loc[:, self.metadata.index]
        y_var = y_var.loc[:, self.metadata.index]
        if self.x_standardization:
            self.x_mean = x.mean(1)
            self.x_std = x.std(1)
            x = x.apply(lambda x: (x-x.mean())/x.std(), axis=1)
        self.data_path = '{}/processed_data'.format(self.output_path)
        os.makedirs(self.data_path, exist_ok=True)
        self.microbe_list = x.index.tolist()
        self.y_std = y.std(1)
        x.to_csv('{}/x.csv'.format(self.data_path))
        y.to_csv('{}/y.csv'.format(self.data_path))
        y_var.to_csv('{}/yVariance.csv'.format(self.data_path))
        self.metadata.to_csv('{}/metadata.csv'.format(self.data_path))

    def _estimate_interactions(self):
        print('Fitting continuous-time regression hidden Markov model...')
        with Pool(processes=self.n_jobs) as pool:
            imap = pool.imap(self._estimate_interactions_for_each_trial, 
                             range(self.n_init))
            _ = list(tqdm(imap, total=self.n_init))

    def _estimate_interactions_for_each_trial(self, t):
        python_source_dir = os.path.abspath(os.path.dirname(__file__))
        ctrhmm_bin_path = os.path.join(python_source_dir, 'bin', 'CTRHMM')
        for K in self.K_list:
            condition = 'K{}'.format(K)
            trial = 'Trial{}'.format(str(t).zfill(len(str(self.n_init))))
            this_output_path = os.path.join(self.output_path, 'results', 
                                            condition, trial)
            os.makedirs(this_output_path, exist_ok=True)
            command = '{} {}/x.csv {}/y.csv {}/yVariance.csv {}/metadata.csv'\
                      ' -k {} -n {} -a {} -o {}'\
                      .format(ctrhmm_bin_path, self.data_path, self.data_path, 
                              self.data_path, self.data_path, 
                              K, self.max_iter, self.tol, this_output_path)
            subprocess.call(command.split(' '),
                            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    def _copy_best_results(self):
        self._derive_maximized_elbos()
        os.makedirs('{}/best_results'.format(self.output_path), exist_ok=True)
        best_K = self.elbo_df.max(1).idxmax()
        condition = 'K{}'.format(best_K)
        best_trial = self.elbo_df.loc[best_K].idxmax()
        trial = 'Trial{}'.format(str(best_trial).zfill(len(str(self.n_init))))
        best_result_path = os.path.join(self.output_path, 'results', 
                                        condition, trial)
        for _file in os.listdir(best_result_path):
            shutil.copy(os.path.join(best_result_path, _file), 
                        '{}/best_results'.format(self.output_path))
        self.best_result_K = len(pd.read_csv('{}/best_results/Q.csv'
                                 .format(self.output_path), header=None).index)

    def _modify_interaction_param(self):
        for k in range(self.best_result_K):
            phi = pd.read_csv('{}/best_results/phi{}.csv'.format(self.output_path, k), header=None)
            phi.index = self.microbe_list + ['Growth']
            phi.columns = self.microbe_list
            modified_phi = (phi.iloc[:-1, :].T / self.x_std).T.copy()
            modified_phi0 = phi.iloc[-1, :] - phi.iloc[:-1, :].T.dot(self.x_mean/self.x_std)
            modified_phi0.name = 'Growth'
            modified_phi = modified_phi.append(modified_phi0)
            modified_phi.to_csv('{}/best_results/interaction_parameters{}.csv'
                                .format(self.output_path, k))

    def _derive_maximized_elbos(self):
        elbo_list_list = []
        for K in self.K_list:
            condition = 'K{}'.format(K)
            elbo_list = []
            for t in range(self.n_init):
                trial = 'Trial{}'.format(str(t).zfill(len(str(self.n_init))))
                this_output_path = os.path.join(self.output_path, 'results', 
                                                condition, trial)
                elbo = pd.read_csv('{}/ELBO.csv'.format(this_output_path), 
                                   header=None).iloc[-1, 0]
                elbo_list.append(elbo)
            elbo_list_list.append(elbo_list)
        self.elbo_df = pd.DataFrame(elbo_list_list, index=self.K_list, 
                                    columns=range(self.n_init))
        self.elbo_df.index.name = 'Kinit\\Trial'
        self.elbo_df.to_csv('{}/KinitTrialELBO.csv'.format(self.output_path))

    def plot(self):
        os.makedirs('{}/figures'.format(self.output_path), exist_ok=True)
        self._plot_gp_regression()
        self._plot_maximized_elbo()
        self._plot_viterbi_path()
        self._plot_interaction_networks()

    def _plot_gp_regression(self):
        S = len(self.model_srs_list)
        D = len(self.model_srs_list[0].index)
        fig = plt.figure(figsize=(5*S, 4*D))
        for s, model_srs in enumerate(self.model_srs_list):
            for m in range(D):
                ax = fig.add_subplot(D, S, m*S + s + 1)
                model_srs.iloc[m].plot(ax=ax)
                if s == 0:
                    ax.set_ylabel(str(model_srs.index.tolist()[m]))
                if m == 0:
                    ax.set_title(self.subject_list[s])
        fig.text(0.5, -0.05, 'Timepoint', ha='center', size=20)
        fig.text(-0.05, 0.5, 'Log quantitative abundance', 
                 va='center', rotation='vertical', size=20)
        plt.tight_layout()
        plt.savefig('{}/figures/gp_regression.pdf'.format(self.output_path))

    def _plot_maximized_elbo(self):
        plt.figure()
        max_maximized_elbo_srs = self.elbo_df.max(1)
        max_maximized_elbo_srs.plot()
        plt.plot(max_maximized_elbo_srs.idxmax(), 
                 max_maximized_elbo_srs.max(), 'rx')
        xlabel = 'The number of states'
        ylabel = 'Maxmized ELBO'
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig('{}/figures/max_maximized_elbo.pdf'.format(self.output_path))
        plt.cla()
        sns.boxplot(data=self.elbo_df.T.melt(var_name=xlabel, value_name=ylabel), 
                    x=xlabel, y=ylabel, whis='range')
        plt.tight_layout()
        plt.savefig('{}/figures/maximized_elbo_boxplot.pdf'.format(self.output_path))

    def _plot_viterbi_path(self):
        from matplotlib import colors as mcolors
        color_list = (list(mcolors.BASE_COLORS.keys()) + 
                      list(mcolors.CSS4_COLORS.keys()))
        z = pd.read_csv('{}/best_results/ViterbiPath.csv'.format(self.output_path), 
                        header=None)
        z.index = self.metadata.index
        z.columns = ['state']
        merged_z = self.metadata.join(z)
        S = len(self.subject_list)
        frn, fcn = None, None
        if S == 1:
            fcn = 1
            frn = 1
        else:
            fcn = 2
            frn = math.ceil(S/fcn)
        fig = plt.figure(figsize=(fcn*10, frn*3.2))
        for s, subject in enumerate(self.subject_list):
            this_merged_z = merged_z[merged_z['subjectID']==subject].copy()
            ax = fig.add_subplot(frn, fcn, s+1)
            plot_state(this_merged_z, self.best_result_K, color_list, subject, ax)
        plt.tight_layout()
        plt.savefig('{}/figures/viterbi_path.pdf'.format(self.output_path))

    def _plot_interaction_networks(self):
        phi = [None] * self.best_result_K
        for k in range(self.best_result_K):
            phi[k] = pd.read_csv(('{}/best_results/phi{}.csv'
                                  .format(self.output_path, k)), 
                                  header=None)
            phi[k].index = self.microbe_list + ['Growth']
            phi[k].columns = self.microbe_list
        adjusted_phi = phi.copy()
        for k, phik in enumerate(adjusted_phi):
            adjusted_phi[k] = phik / self.y_std
        frn, fcn = None, None
        if self.best_result_K == 1:
            fcn = 1
            frn = 1
        else:
            fcn = 2
            frn = math.ceil(self.best_result_K/fcn)
        fig = plt.figure(figsize=(10*fcn, 9*frn))
        plt.subplots_adjust(hspace=0.1)
        for k, phik in enumerate(adjusted_phi):
            ax = fig.add_subplot(frn, fcn, k+1)
            plot_directed_network(phik, ax, 'State{}'.format(k+1))
        plt.tight_layout()
        plt.savefig('{}/figures/interaction_networks.pdf'
                    .format(self.output_path))

    def _return_member_for_debug(self):
        return self.elbo_df
