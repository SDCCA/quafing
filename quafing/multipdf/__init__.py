import os

from .factorized_multi_dimensional_pdf import FactorizedMultiDimensionalPDF as FactorizedMDPDF

multipdf_types = {
	'factorized': FactorizedMDPDF
}

def multipdf_init(pdftype):
    _check_type(pdftype)
    multipdf = multipdf_types[pdftype]
    multipdf._type = pdftyp
    return multipdf

def _check_type(pdftype):
    if pdftype not in multipdf_types:
        raise NotImplementedError(
        	"Multi-dimensional PDF type %s unknown. Supported types are: %s" % (pdftype, ', '.join(multipdf_types.keys())))	