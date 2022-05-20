""" IO Handler for Excel/Spreadsheet file format """
import pandas as pd

from quafing.io.base_io_handler import IOHandler

class ExcelHandler(IOHandler):
	""" Class for IO of data in Excel/spreadsheet format (e.g. .xls) format """

	def read(self, header_row=2, question_row=1, type_row=0, sheet=0, start_row=3, skip_at_end=0, **kwargs)
	    """
	    read data and metadata from a Excel/spreadsheet type file into memory

	    :param header_row: number of row (0-indexed) containing column names
	    :param question_row: number of row (0-indexed) containing number of question
	    :param type_row: number of row (0-indexed) containing column type definitions 
	    :param sheet: number or name of sheet
	    :param start_row: row number where data begins
	    :param skip_at_end: number of rows at end to skip
	    :param kwargs: optional addittional keyword arguments of pandas 'read_excel' method 

	    :return data: parsed data in DataFrame
	    :return metadata: dict of arrays with column metadata  
	    """

	    mdargs = {'question_row':question_row,
        			'type_row':type_row
        }

        pdargs = {
        			'header':header_row,
        			'sheet_name':sheet,
        			'skiprows':start_row,
        			'skipfooter':skip_at_end
        }

        pdargs.update(kwargs)

        metadata = read_metadata(**mdargs,**pdargs)
        data = read_data(metadata['ColNames'],**pdargs)
        return metadata, data

    def read_metadata(self,**kwargs1,**kwargs2)
        """
        """

        ColNameKWArgs = {'sheet_name':kwargs['sheet_name'],'header':kwargs['header'],'nrows':0}
        ColTypeKWArgs = {'sheet_name':kwargs['sheet_name'],'header':None,'nrows':1,'skiprows':kwargs['type_row']}
        QuestionNumbersKWArgs = {'sheet_name':kwargs['sheet'],'header':None,'nrows':1,'skiprows':kwargs['question_row']}

        ColNames = pd.read_excel(self.path, **ColNameKWArgs).to_numpy()
        ColTypes = pd.read_exel(self.path, **ColTypeKWArgs).values[0]
        QuestionNumbers = pd.read_exel(self.path, QuestionNumbersKWArgs).values[0]
        metadata = {'ColNames':ColNames ,'ColTypes':ColTypes, 'QuestionNumbers':QuestionNumbers}
        return metadata 

    def read_data(self,ColNames,**kwargs)
        """
        """

        kwargs.update({'names':ColNames,'header':None})

        data = pd.read_excel(self.path, **kwargs)
        return data

    def write(self)
        """
        write function

        TODO
        """
        raise NotImplementedError(
    	    "Class %s doesn't implement write() yet"% self.__class__.__name__)