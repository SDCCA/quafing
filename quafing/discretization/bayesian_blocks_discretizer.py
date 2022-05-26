""" discretizer using bayesian histogram optimizatiton """
import os
import numpy as np
import pandas as pd
from astropy.stats import bayesian_blocks

from quafing.discretization.base_discretizer import Discretizer

class BayesianBlockDiscretizer(Discretizer):
	"""
	class to perfom discretization of 1 dimensional ccontinuous
	real valued data using bayesian optimized histogram binning
	"""

	def perform_discretization(self, *args, *kwargs):
		"""
		create discretization of data object

		:param *args: optional: arguments to pass to methods used
		:param *kwargs: optional: keyword arguments to pass to methods used
    	:return bins_list: discretization. List of arrays with bin bordeers
    	"""
        """
    	if self._dim > 1:
    		raise RuntimeError(
    			'Dimensionality of data is greater than 1 and too large for method')
    	"""

    	cont_cols = [c['ColNames'] for c in self._colmetadata if c['ColType'] == 'c']
        cont_data = self._data[cont_cols]

        bins_list = []
        for i in range(len(cont_data.columns)):
            d = cont_data.iloc[:,i] 
            bb = bayesian_blocks(d)
            bins_list.append(bb)

        return bins_list
