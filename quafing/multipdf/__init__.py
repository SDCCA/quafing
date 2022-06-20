import os

from .factorized_multi_dimensional_pdf import FactorizedMultiDimensionalPDF as FatorizedMDPDF

mdpdfTypes = {
	'factorized': FatorizedMDPDF
}

def obtain_mdpdf(data,colmetadata,pdftype,*args,**kwargs):
    _check_type(pdftype)
    multipdf = mddpdftypes[pdftype]
    return multipdf(data,colmetadata,*args,**kwargs)

def _check_type(pdftype):
    if pdftype not in multipdf_types:
        raise NotImplementedError(
        	"Multi-dimensional PDF type %s unknown. Supported types are: %s" % (pdftype, ', '.join(multipdf_types.keys())))	