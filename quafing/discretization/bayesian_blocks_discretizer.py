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

    def perform_discretization(self,*, cols=['c'], byType=True, **kwargs):
        """
        create discretization of data object

        :param cols: keyword argument specifying list of columns to discretize. list of types, or names or indicees
        :param byType: keywword specifying wheter column specification is based on type  or name/index
        :param **kwargs: optional: keyword arguments to pass to methods used
        :returns discreiztion: discretization (bin list). List of dicts with column names and array with bin borders, None if not discretized.
        """
        


        
        if byType:
            types = list(set([c['ColTypes'] for c in self._colmetadata]))
            if all([coltype in types for coltype in cols]):
                colnames = [ c['ColNames'] for c in self._colmetadata if c['ColTypes'] in cols]
            else:
                raise ValueError(
                    'specified column types not all present in data') 
        else:
            if all([isinstance(cols[i],str) for i in range(len(cols))]):
                colnames = cols
            elif all([isinstance(cols[i],int) for i in range(len(cols))]):
                colnames = [self._colmetadata[cols[i]]["ColNames"] for i in range(len(cols))]
            else:
                raise ValueError(
                    'Column specification contains mixed types')

        discretization = []
        for i in range(len(self._colmetadata)):
            if self._colmetadata[i]['ColNames'] in colnames:
                d = self._data.loc[:,self._colmetadata[i]['ColNames']]
                bb = bayesian_blocks(d,**kwargs)
                discretization.append({'ColNames':self._colmetadata[i]['ColNames'], 'Disc':bb})
            else:
                 discretization.append({'ColNames':self._colmetadata[i]['ColNames'], 'Disc':None})

        return discretization
