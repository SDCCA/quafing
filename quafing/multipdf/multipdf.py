from quafing.multipdf import obtain_mdpdf

def create_multi_pdf(data,colmetadata,mdpdfType,*args,**kwargs):
	mdpdf = obtain_mdpdf(data,colmetadata,*args,**kwargs)
	return mddpdf