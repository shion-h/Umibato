#
# _gpmodule.py
#
# Copyright (c) 2020 Shion Hosoda
#
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php
#

import logging
import warnings
import GPy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
logging.getLogger().setLevel(logging.WARNING)
warnings.filterwarnings('ignore')
sns.set()


# _kernel = GPy.kern.Matern32(input_dim=1)
# _kernel = GPy.kern.Matern52(input_dim=1)
_kernel = GPy.kern.RBF(input_dim=1, variance=1, lengthscale=1)


def fit_gp_model(x, y, init_kernel=_kernel, sig_level=10.0):
    # x: pandas.Series
    # y: pandas.Series

    kernel = init_kernel.copy()
    # inference_method = GPy.inference.latent_function_inference.Laplace()
    model = GPy.models.GPRegression(x[:,None], y[:,None], kernel=kernel)
    # model = GPy.core.GP(X=x[:,None], Y=y[:,None], likelihood=GPy.likelihoods.StudentT(), kernel=kernel, inference_method=inference_method)
    model.optimize()
    if sig_level is not 0.0:
        upper_and_lower = model.predict_quantiles(x[:,None], quantiles=(sig_level/2.0, 100.0-sig_level/2.0))
        lower_bound = upper_and_lower[0]
        upper_bound = upper_and_lower[1]
        isnot_outlier = (y[:,None] > lower_bound) & (y[:,None] < upper_bound)
        isnot_outlier = isnot_outlier.reshape(-1)
        kernel = init_kernel.copy()
        model = GPy.models.GPRegression(x[isnot_outlier,None], y[isnot_outlier,None], kernel=kernel)
        # model = GPy.core.GP(X=x[isnot_outlier,None], Y=y[isnot_outlier,None], likelihood=GPy.likelihoods.StudentT(), kernel=kernel, inference_method=inference_method)
        model.optimize()
    return(model)

def estimate_grad_numerically(model, x, epsilon=1.0e-5):
    # x.shape = (D, 1)
    x_plus_0 = model.predict_noiseless(x + epsilon)[0]
    x_minus_0 = model.predict_noiseless(x - epsilon)[0]
    return (x_plus_0 - x_minus_0) / (2*epsilon)

def estimate_grad_variance(model, timepoints, L=100, epsilon=1.0e-5):
    t_plus_0 = timepoints + epsilon
    t_minus_0 = timepoints - epsilon
    t = np.sort(np.append(t_plus_0, t_minus_0))
    f_samples = model.posterior_samples_f(t.reshape(-1, 1), full_cov=True, size=L)
    f_samples = f_samples.reshape(1, -1, L)[0]
    gradients = []
    for i in range(len(timepoints)):
        tmp = (f_samples[2 * i] - f_samples[2 * i + 1]) / (2 * epsilon)
        gradients.append(tmp)
    gradients = np.array(gradients)
    return np.var(gradients, ddof=1, axis=1)
