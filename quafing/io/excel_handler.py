""" IO Handler for Excel/Spreadsheet file format """
import pandas as pd

from quafing.io.base_io_handler import IOHandler

class ExcelHandler(IOHandler):
	""" Class for IO of data in .xls format """

	def read(self, **kwargs)
	    """
	    read data and metadata from a Excel/spreadsheet type file into memory

	    :param header_row: number of row (0-indexed) containing column names
	    :param question_row: number of row (0-indexed) containing number of question
	    :param type_row: number of row (0-indexed) containing column type definitions 
	    :param sheet: number(s) or name(s) of sheet(s)
	    :param start_row: row number where data begins
	    :param skipfooter: number of rows at end to skip
	    :param kwargs: optional addittional keyword arguments of pandas 'read_excel' method 

	    :return data: parsed data in DataFrame
	    :return metadata: dict of arrays with column metadata  
	    """

        defaults = {
        			'header_row':2,
        			'question_row':1,
        			'type_row':0,
        			'sheet':0,
        			'start_row':3,
        			'skipfooter':0
        }

        defaults.update(kwargs)




    def read_metadata(self,**kwargs)
        """
        """

        ColNames = pd.read_excel(self.path, sheet_name=kwargs['sheet'],header=kwargs['header_row'],nrows=0).tonumpy()
        ColTypes = pd.read_exel(self.path, sheet_name=kwargs['sheet'],header=None,nrows=1,skiprows=kwargs['type_row']).values[0]
        QuestionNumbers = pd.read_exel(self.path, sheet_name=kwargs['sheet'],header=None,nrows=1,skiprows=kwargs['question_row']).values[0]
        metadata = {'ColNames':ColNames ,'ColTypes':ColTypes, 'QuestionNumbers':QuestionNumbers}
        return metadata



    def read_data(self,)
        """
        """

        