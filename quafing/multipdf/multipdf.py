from quafing.multipdf import multipdf_init
from quafing.multipdf.multipdf_collection import MultiPdfCollection

def create_multi_pdf(mdpdfType, data,colmetadata,calculate=True,*args,**kwargs):
	"""
	create a multipdf object from data and (column)metadata

	:param mdpdfType: str; Type of multidimensional pdf to create. See quafing.multipdf.__init__() for valid types
	:param data: pandas DataFrame; (columnar) data from whhih to create multipdf. 
	:param colmetadta: array of dictionaries; metadata for each column
	:param calculate: bool; keyword to trigger calculation of (component) pdfs for created multipdf object. default True
	:param args: optional; arguments to be  passed to multi_pdf class calculate_pdf() function
	:param kwargs: optional; keyword arguments to be passed to multi_pdf class calculate_pdf() function
	:return mdpdf: MultiDimensionalPdf object of specified type  
	"""
	mdpdf = multipdf_init(mdpdfType)
	mdpdf._import_data(data,colmetadata)
	mdpdf._basic_validation()
	if calculate:
	    mdpdf.calculate_pdf(*args,**kwargs)  
	return mdpdf


def create_mdpdf_collection(mdpdfType, group_data, group_labels,colmetadata, calculate=True, validate_metadata=False, *args, **kwargs):
    """
    TODO
    """
    mdpdfs = []
    for i, data in enumerate(group_data):
        mdpdf = create_multi_pdf(mdpdfType,data, colmetadata, calculate=calculate, *args, **kwargs)
        mdpdfs.append(mdpdf)
    mdpdf_collection = MultiPdfCollection(mdpdfs,group_labels, colmetadata, mdpdfType, validate_metadata=validate_metadata)
    return mdpdf_collection