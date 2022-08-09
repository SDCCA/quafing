""" density estimator for discrete 1-d data """

import os
import pandas as pd
import numpy as np

from quafing.density.base_density_estimators import DiscreteDensityEstimator

class DiscreteDensityEstimator1d(DiscreteDensityEstimator):


    def _check_input_1d(self):
        if isinstance(self._data,pd.Series):
            pass    
        elif isinstance(self._data,np.ndarray):
            if self._data.ndim == 1:
                pass
            else:
                raise RuntimeError(
                    'input data is not one dimensional.')
        else:
            raise TypeError(
                'input data is not of supported type (pandas.Series or numpy.ndarray (1d)')

    def obtain_density(self, discrete=True, discretization=None):
        super()._check_discretization_info(discrete=discrete, discretization=discretization)
        self._check_input_1d()

        if self._discrete:
            self._intrinsic_discrete_data_density()
        else:
            self._discretized_data_density()
        return self._discrete_pdf

    def _intrinsic_discrete_data_density(self):
        if isinstance(self._data, pd.Series):
            self._unique = self._data.unique()
        else:
            self._unique = np.unique(self._data)
        disc_pdf = {}
        for val in self._unique:
            count = np.sum(self._data == val)
            disc_pdf[val] = count / len(self._data)
        self._discrete_pdf = disc_pdf

    def _discretized_data_denity(self):
        # Compute the probabilities of each bin
        h = np.histogram(self._data, bins=self._discretization, normed=True)[0]
        widths = [bins[i+1] - bins[i] for i in range(len(bins)-1)]
        probs = h * widths
        # The 'value' of each key is the bin number
        disc_pdf = {}
        for i in range(len(bins)-1):
            disc_pdf[i] = probs[i]
        self._discrete_pdf = disc_pdf

