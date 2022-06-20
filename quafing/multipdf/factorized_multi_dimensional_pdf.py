""" class for a factorized (indvidual independent variables) multidimensional pdf """
import os

import pandas as pd
import numpy as np

from quafing.multipdf.multi_dimensional_pdf import MultiDimensionalPDF
from quafing.density.estimate_density import get_density_estimate

class FactorizedMultiDimensionalPDF(MultiDimensionalPDF):
    """ class for factorizable multi-dimensional PDFs. Variables/dimensions are
    iid, allowing each to be treated separately as 1-dimensional pdfs
    """

    def __init__(self, data, colmetadata, discretization=None):
    	super().__init__(data,colmetadata)

        if 'Disc' not in self._colmetadata[0].keys():
            if discretization is None:
               c.update({'Disc':None}) for c in self._colmetadata
            else:
                for c in self._colmetadata:
                    coldisc= [d for i,d in enumerate(discretization) if d['ColNames'] == c['ColNames']][0]
                    c.update(coldisc)
        else:
            if discretization is None:
                pass
            else:
                for c in self._colmetadata:
                    coldisc= [d for i,d in enumerate(discretization) if d['ColNames'] == c['ColNames']][0]
                    c.update(coldisc)


        fpdfs = []
        for column in self._colmetadata:
            d = self._data[column['ColNames']]
            pdf = get_density_estimate(d,column,method='Discrete1D',discrete=column['discrete'],discretization=column['Disc'])
            fpdfs.append(pdf)

        self._pdfs = fpdfs


