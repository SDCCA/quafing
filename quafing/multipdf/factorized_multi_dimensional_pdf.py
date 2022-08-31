""" class for a factorized (indvidual independent variables) multidimensional pdf """
import os
import warnings

import pandas as pd
import numpy as np

from quafing.multipdf.multi_dimensional_pdf import MultiDimensionalPDF
from quafing.density.estimate_density import get_density_estimate


class FactorizedMultiDimensionalPDF(MultiDimensionalPDF):
    """ class for factorizable multi-dimensional PDFs. Variables/dimensions are
    iid, allowing each to be treated separately as 1-dimensional pdfs
    """

    def calculate_pdf(self,method=None,discretization=None):
        """
        construct multidimensional pdf as individual 1d pdfs per column, using density estimation method
        specified (either passed with column metadata or explicity specified in call) and optional dicretization 
        (either passed with column metadata or explicity specified in call specified. 
        Method must be a valid supported option as listed in quafing.multipdf.__init__

        :param method: str, optional; density estimation method to be used.
        :param discretization: optional; discretization to be used for continuous data for which
                              a discrete pdf is to be returned. Expected format is list of dictionaries
                              as returned by discretize method.
        """
        
        if self._pdf is None:
            if 'Disc' not in self._colmetadata[0].keys():
                if discretization is None:
                    for c in self._colmetadata:
                        c.update({'Disc':None}) 
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
                        warning.warn(
                            f'density method for ALL columns being set to {method}')
                        for c in self._colmetadata:
                            c.update({'density_method':method})
                    else:
                        for c in self._colmetadata:
                            colmeth = [m for i,m in enumerate(method) if m['ColNames'] == c['ColNames']]
                            if len(colmeth) == 0:
                                warning.warn(f"No density method specified for {c['ColNames']}")
                                c.update({'density_method':None})
                            else:
                                c.update(colmeth[0])
            else:
                if method is None:
                    pass
                else:
                    warning.warn(
                        'exisiting density methods are being overwritten')
                    if isinstance(method,str):
                        for c in self._colmetadata:
                            c.update({'density_method':method})
                    else:
                        for c in self._colmetadata:
                            colmeth = [m for i,m in enumerate(method) if m['ColNames'] == c['ColNames']]
                            if len(colmeth) == 0:
                                warning.warn(f"No density method specified for {c['ColNames']}")
                                c.update({'density_method':None})
                            else:
                                c.update(colmeth[0])

            fpdfs = []
            fpdfs_meta = []
            for column in self._colmetadata:
                d = self._data[column['ColNames']]
                pdf = get_density_estimate(d,method=column['density_method'],metadata=None,discrete=column['discrete'],discretization=column['Disc'])
                pdf_meta = self._generate_pdf_meta_entry(column['density_method'], column['ColNames'], len(d.shape), column['discrete'],column['Disc'])
                fpdfs.append(pdf)
                fpdfs_meta.append(pdf_meta)

            self._pdf = fpdfs
            self._pdf_meta = fpdfs_meta
        else:
            pass
