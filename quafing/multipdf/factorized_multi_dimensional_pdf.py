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

    def calculate_pdf(self,method=None,discretization=None):
        
        if self._pdf is None:
            if 'Disc' not in self._colmetadata[0].keys():
                if discretization is None:
                   [c.update({'Disc':None}) for c in self._colmetadata]
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

            if 'density_method' not in self._colmetadata[0].keys():
                if method is None:
                    raise RuntimeError(
                        'no method for density estimation specified')
                else:
                    if isinstance(method,str):
                        [c.update({'density_method':method}) for c in self._colmetadata]
                    else:
                        for c in self._colmetadata:
                            colmeth = [m for i,m in enumerate(method) if m['ColNames'] == c['ColNames']][0]
                            c.update(colmeth)
            else:
                if method is None:
                    pass
                else:
                    if isinstance(method,str):
                        [c.update({'density_method':method}) for c in self._colmetadata]
                    else:
                        for c in self._colmetadata:
                            colmeth = [m for i,m in enumerate(method) if m['ColNames'] == c['ColNames']][0]
                            c.update(colmeth)

            fpdfs = []
            fpdfs_meta = []
            for column in self._colmetadata:
                d = self._data[column['ColNames']]
                pdf = get_density_estimate(d,method=column['density_method'],metadata=None,discrete=column['discrete'],discretization=column['Disc'])
                pdf_meta = {'method':column['density_method'], 'data_labels':column['ColNames'], 'data_dimension':d.shape, 'discrete':column['discrete'],'discretiztion':column['Disc']}
                fpdfs.append(pdf)
                fpdfs_meta.append(pdf_meta)

            self._pdf = fpdfs
            self._pdf_meta = fpdfs_meta
        else:
            pass
