""" class for a factorized (indvidual independent variables) multidimensional pdf """
import os

import pandas as pd
import numpy as np

from quafing.multipdf.multi_dimensional_pdf import MultiDimensionalPDF
from quafing.discretization import discretize
from quafing.density.estimate_density import estimate_density

class FactorizedMultiDimensionalPDF(MultiDimensionalPDF):
    """ class for factorizable multi-dimensional PDFs. Variables/dimensions are
    iid, allowing each to be treated separately as 1-dimensional pdfs
    """

    def __init__(self, data, colmetadata, discretization=None):
        super().__init__(data,colmetadata)


