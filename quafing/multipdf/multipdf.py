from quafing.multipdf import multipdf_init

def create_multi_pdf(mdpdfType, data,colmetadata,calculate=True,*args,**kwargs):
	mdpdf = multipdf_init(mdpdfType)
	mdpdf._import_data(data,colmetadata)
	mdpdf._basic_validation()
	if calculate:
	    mdpdf.calculate_pdf(*args,**kwargs)  
	return mdpdf