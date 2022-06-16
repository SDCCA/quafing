""" pre processing steps for loaded questionnaaire data """
import pandas as pd
import warnings

from quafing.discretization.discretize import discretize

class PreProcessor(object):
    """ class for prepocessing raw data.
    """
    def __init__(self,rawdata, rawmetadata):
        """
        perform checks on rawdata and rawmetadata

        :param rawdata: quesionnaire data in pandas DataFrame format
        :param rawmetadata: quesstionnaire metadata. Dict of arrays Colnames, ColTypes, 
        QuestionNumbers, of same length as number of columns in rawdata 
        """
       
        self._rawdata = rawdata 
        self._rawmetadata = rawmetadata
        self._rawcolmetadata = None
        self._data = None
        self._colmetadata = None
        self._groupingcolumn = None
        self._grouplabels = None
        self._groups = None
        
        """ validate input formaat """
        if not isinstance(rawdata, pd.DataFrame):
            raise RuntimeError(
                'raw data input is not of type pandas DataFrame')

        """check rawmetadata keys"""
        for key in ["ColNames","ColTypes"]:
            if self._check_key(key, self._rawmetadata):
                continue
            else:
                raise ValueError(
                    "Required key %s does not exist in metadata dictionary"%key)

        if self._check_key("QuestionNumbers",self._rawmetadata):
            pass
        else:
            warnings.warn("Warning: key 'QuestionNumbers' does not exist in metadata dictionary")

        self._check_dimensions()

        self._generate_col_metadata()

    def _check_key(self, key, dd):
        """
        check whether key is present in dictionary

        :param key:
        :param dd: dictionary
        :return iskey: Boolean indicating whether key exists in dictionary
        """
        iskey = key in dd.keys()
        return iskey

    def _check_dimensions(self):
        """
        check whether data and metadata dimensions match as expected
        """
        mdKeys = list(self._rawmetadata.keys())
        mdLength = len(self._rawmetadata[mdKeys[0]])

        for key in mdKeys:
            if len(self._rawmetadata[key]) != mdLength:
                raise ValueError(
                    "mismatch in length of metadata fields")

        if mdLength != len(self._rawdata.columns):
            raise ValueError(
                "Mismatch in length between data and metadata")

    def _generate_col_metadata(self):
        """
        create column wise meta data dictionary"
        """
        ColTypes = self._rawmetadata['ColTypes']
        ColNames = self._rawmetadata['ColNames']
        if 'QuestionNumbers' in self._rawmetadata.keys():
            QuestionNumbers = self._rawmetadata['QuestionNumbers']
        else:
            QuestionNumbers = [None]*len(ColTypes)
        colmetadata = []
        for ColTypes,ColNames,QuestionNumbers in zip(ColTypes,ColNames,QuestionNumbers):
               colmetadata.append({'ColTypes':ColTypes,
                                'ColNames':ColNames,
                                'QuestionNumbers':QuestionNumbers})

        self._rawcolmetadata = colmetadata

    def select_columns(self,cols,deselect=False,by_type=True,select_all=False):
        """
        select columns for use in analysis. Selection can be based on column name 
        or number (0-indexed) or on column type as provided in metadata. 
        If deselect is set, the complement of the specified columns is seleted.
        select_all provides a convenince function to select all coluumns.

        :param cols: list or array of column names or numbers (0-indexed), or list of types
        :param deslect: keyword to select complement 
        :param by_type: keyword to set selection by column type
        :param select_all: keyword to simply select all data columns
        """
        if select_all == True:
            self._data = self._rawdata.copy() 
            self._colmetadata = self._rawcolmetadata.copy()
        else:
            if by_type == True:
                self._select_by_type(cols,deselect=False)
            else:
                self._select_by_label(cols,deselect=False)

    def _check_selection(self):
        if (self._data is None) or (self._colmetadata is None):
            raise RuntimeError(
                'No data has been selected for analysis. Please select data using the select_columns() method.')

    def _select_by_type(self,cols,deselect=False):
        """
        select columns for analysis by column type

        :param cols: list of column types (as defined in metadata) to (de)select
        :param deselect: keyword to select complement
        """
        cnames = []
        colmetadata = []
        if deselect:
            for i in len(self._rawcolmetadata):
                if self._rawcolmetadata[i]["ColTypes"] in cols:
                    cnames.append(self._rawcolmetadata[i]["ColNames"])
                else:
                    colmetadata.append(self._rawcolmetadata[i])
        else:
            for i in len(self._rawcolmetadata):
                if self._rawcolmetadata[i]["ColTypes"] not in cols:
                    cnames.append(self._rawcolmetadata[i]["ColNames"])
                else:
                    colmetadata.append(self._rawcolmetadata[i])

        self._data = self._rawdata.copy().drop(columns=cnames)
        self._colmetadata = colmetadata



    def _select_by_label(self,cols,deselect=False):
        """
        select columns for analysis by name or index

        :param cols: list of names (strings) or indices (integers, 0-indexed) to (de)select
        :param deselect: keyword to select complement 
        """
        if all([isinstance(cols[i],str) for i in range(len(cols))]):
            colsnames = cols
        elif all([isinstance(cols[i],int) for i in range(len(cols))]):
            colsnames = [self._rawcolmetadata[cols[i]]["ColNames"] for i in range(len(cols))]
        else:
            raise ValueError(
                'Column specification contains mixed types')

        if deselect:
            cnames = colsnames
            colmetadata = [self._rawcolmetadata[i] for i in range(len(self._rawcolmetadata)) if self._rawcolmetadata[i]["ColNames"] not in colsnames]

        else:
            cnames = [self._rawcolmetadata[i]["ColNames"] for i in range(len(self._rawcolmetadata)) if self._rawcolmetadata[i]["ColNames"] not in colsnames]
            colmetadata = [self._rawcolmetadata[i] for i in range(len(self._rawcolmetadata)) if self._rawcolmetadata[i]["ColNames"] in colsnames]

        self._data = self._rawdata.copy().drop(columns=names)
        self._colmetadata = colmetadata

    def shuffle(self):
        """
        shuffle data rows
        """
        self._check_selection()
        self._data=self._data.sample(frac=1)

    def randomize_data(self,col):
        """
        !!! CAUTION !!!
        Randomize the data set by reassigning the labels to be used for grouping.
        This process is IRREVERSSIBLE on the selected data set. Recovering
        non-randomized data requires reselection from the raw data.

        :param col: name (str) or index (int, 0-indexed, based on self._data) of column to group by 
        """
        self._check_selection()
        if isinstance(col,str):
            if col in self._data.columns:
                gbcol = col
            else:
                raise ValueError(
                    'Column name specified for grouping does not exist in selected data (self._data)')
        elif isinstance(col,int):
            gbcol = self._data.columns[col]

        labels = self._data.loc[:,gbcol].copy()
        print(labels.sample(frac=1))
        self._data.loc[:,gbcol] = labels.sample(frac=1)


    def group(self,col):
        """
        group rows by value of specified column.
        col can be an index (int) or a column name (str)

        :param col: name (str) or index (int, 0-indexed, based on self._data) of column to group by
        :param returncol: keyword to return name of column that was ussed for grouping. Useful when index was specifed
        :return gbcol: Optional: name (str) of column that was grouped by
        """
        self._check_selection()
        if isinstance(col,str):
            if col in self._data.columns:
                gbcol = col
            else:
                raise ValueError(
                    'Column name specified for grouping does not exist in selected data (self._data)')
        elif isinstance(col,int):
            gbcol = self._data.columns[col]

        self._data.groupby(by=gbcol)
        self._groupingcolumn = gbcol
        self._grouplabels = self._data[gbcol].unique()
        

    def split_to_groups(self, col):
        """
        split selected data into groups

        :param col: name (str) or index (int, 0-indexed, based on self._data) of column to group by
        """
        self._check_selection()
        if self._grouplabels == None:
            self.group(col)

        groups = []
        for i in range(len(self._grouplabels)):
            groups.append(self._data.loc[self._data[self._groupingcolumn] == self._grouplabels[i]])

        self._groups = groups

    def get_joint_discretization(self,method=None,*args,**kwargs):
        """
        Obtain discretization of columns/questions with continuous answer space,
        common across all groups being considered

        :param method: keyword specifying discretization method. See discretization documentation
        :param *args: optional arguments to be passed to discretization methods
        :param *kwargs: optional keyword arguments to be passed to discrettization methods
        :return discretization: list of arrays ith biin borders
        """

        self._check_selection()
        disc = discretize(self._data,self._colmetadata,method=method,*args,**kwargs)
        return disc 
