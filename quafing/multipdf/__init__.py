import os

from .factorized_multi_dimensional_pdf import FactorizedMultiDimensionalPDF as FactorizedMDPDF

multipdf_types = {
    'factorized': FactorizedMDPDF
}

def multipdf_init(pdftype):
    """
    Initialize a multi-dimensional pdf object of specified type

    :param pdftype: str, Type of multi dimensional pdf to initialize. One of keys in multipdf_types (above)
    :return multipdf: initialized instance of requested multi-dimensional pdf object. _type attribute set
    """
    _check_type(pdftype)
    multipdf = multipdf_types[pdftype]()
    multipdf._type = pdftype
    return multipdf

def _check_type(pdftype):
    """
    check whether multi-dimensional pdf type is supported

    :param pdftype: str, type of multi-dimensional pdf object to initializee use. One of valid keys in multipdf_types 
    """
    if pdftype not in multipdf_types:
        raise NotImplementedError(
            "Multi-dimensional PDF type %s unknown. Supported types are: %s" % (pdftype, ', '.join(multipdf_types.keys())))	