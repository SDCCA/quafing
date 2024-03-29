""" pre processing steps for loaded questionnaire data """
import pandas as pd
import numpy as np
import copy
import warnings

from quafing.discretization.discretize import discretize
from quafing.density import _check_density_method 

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
        self._groupcolmetadata = None
        
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

    def select_columns(self,cols=None,deselect=False,by_type=True,select_all=False):
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
                self._check_colTypes(cols)
                self._select_by_type(cols,deselect=deselect)
            else:
                self._check_colNames(cols)
                self._select_by_label(cols,deselect=deselect)

    def _check_selection(self):
        """
        check whether data for has already been selected from raw input data
        """
        if (self._data is None) or (self._colmetadata is None):
            raise RuntimeError(
                'No data has been selected for analysis. Please select data using the select_columns() method.')

    def _check_colTypes(self, cols):
        """
        check whether specified column type(s) exist in raw input (meta)data

        :param cols: (list of) string(s) specifiying column types
        """
        for i in range(len(cols)):
            if cols[i] not in [d['ColTypes'] for d in self._rawcolmetadata]:
                warnings.warn("Warning: Column type '%s' not found in data. Please check column type."%cols[i])


    def _check_colNames(self, cols):
        """
        check whether specified column name(s) exist in raw input (meta)data

        :param cols: (list of) string(s) specifiying column names
        """
        for i in range(len(cols)):
            if cols[i] not in [d['ColNames'] for d in self._rawcolmetadata]:
                warnings.warn("Warning: Column name %s not found in data. Please check column name."%cols[i])
        

    def _select_by_type(self,cols,deselect=False):
        """
        select columns for analysis by column type

        :param cols: list of column types (as defined in metadata) to (de)select
        :param deselect: keyword to select complement
        """
        cnames = []
        colmetadata = []
        if deselect:
            for i in range(len(self._rawcolmetadata)):
                if self._rawcolmetadata[i]["ColTypes"] in cols:
                    cnames.append(self._rawcolmetadata[i]["ColNames"])
                else:
                    colmetadata.append(self._rawcolmetadata[i])
        else:
            for i in range(len(self._rawcolmetadata)):
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

        self._data = self._rawdata.copy().drop(columns=cnames)
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

        labels = self._data.loc[:,gbcol].copy().sample(frac=1)
        self._data.loc[:,gbcol] = labels.values

    def _validate_by_label(self,cols):
        """
        Convert column specification by label (i.e. column index or name) to column name.
        verify that specifiation supplied uses either numerical index OR name.

        :param cols: (list of) column name(s) (str) or indicees (int, 0-indexed, based on self._data)
        :return colnames: list of column names    
        """
        colnames=None
        if all([isinstance(col,str) for _,col in enumerate(cols)]):
            colnames = cols
        elif all([isinstance(col,int) for _,col in enumerate(cols)]):
            colnames = [self._colmetadata[cols[i]]["ColNames"] for i in range(len(cols))]
        else:
            raise ValueError(
                    'Column specification contains mixed types')
        return colnames

    def set_cont_disc(self,cols=['c'],*,by_type=True,complement=True, disccols=[]):
        """
        label columns as containing continuous or discrete data. required for density estimation.
        Updates column metadata 

        :param cols: list of columns with continuous data. columns can be specified either by type
                     (requires by_type=True), or by reference to column name or 0-based index
                     (based on self._data). 
        :param by_type: keyword, bool (default True); if true cols is interpreted a list of strings specifying
                    column types) containing continuous data. If False, cols is interpreted as a list
                    of columns, specified either by name or by index 
        :param complement: keyword, bool (default=False), optional; If True, all columns not specified in cols
                         are set to contain discrete data. If False, columns not specified in cols will have None
                         as their ccolumn meta data discrete value, unless specied as discrete in the optional
                         disccols parameter
        :param disccols: keyword, optional; list of columns with continuous data. columns can be specified either by type
                     (requires by_type=True), or by reference to column name or 0-based index (based on self._data).
        """

        self._check_selection()
        if not by_type:
            cols = self._validate_by_label(cols)
            if len(disccols) != 0:
                disccols = self._validate_by_label(disccols)

        if by_type:
            moniker = 'ColTypes'
        else:
            moniker = 'ColNames'
        
        for c in self._colmetadata:
            if c[moniker] in cols:
                print (c[moniker])
                disc_entry = {'discrete':False}
            else:
                if complement:
                    disc_entry = {'discrete':True}
                else:
                    if c[moniker] in disccols:
                        disc_entry = {'discrete':True}
                    else:
                        disc_entry = {'discrete':None}
            c.update(disc_entry)                

    def set_density_method(self,method=None,cols=['c'],*,by_type=True):
        """
        Specify density estimation method to be used (for groups of) columns
        Updates column metadata

        :param method: str, or list of str, or dict with colnames as keys and method a value.
                        If method is a list or a dict it MUST be equal in length to cols
        :param cols: list of columns data. columns can be specified either by type
                     (requires by_type=True), or by reference to column name or 0-based index
                     (based on self._data).
        :param by_type: keyword, bool (default True); if true cols is interpreted a list of strings specifying
                    column types). If False, cols is interpreted as a list of columns, specified either by name 
                    or by index.              
        """

        self._check_selection()

        if isinstance(method,str):
            _check_density_method(method)
            if not by_type:
                cols = self._validate_by_label(cols)
        elif isinstance(method, dict) or isinstance(method,list):
            if isinstance(method,dict):
                cols = []
                tmethod =[]
                for k,m in method.items():
                    cols.append(k)
                    tmethod.append(m)
                method = tmethod
                del(tmethod)
            if not by_type:
                cols=self._validate_by_label(cols)
            if len(method) != len(cols):
                    raise RuntimeError(
                        'number of specified methods dwoes not match number of specified columns/types')
            else:
                for m in method:
                    _check_density_method(m)
        else:
                raise RuntimeError(
                    'no method for density estimation specified')

        if by_type:
            moniker = 'ColTypes'
        else:
            moniker = 'ColNames'

        for c in self._colmetadata:
            c.update({'density_method':None})
            if isinstance(method,str):
                if c[moniker] in cols:
                    c['density_method'] = method
            else:
                for i, m in enumerate(method):
                    if c[moniker] == cols[i]:
                        c['density_method']= m

        

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

        self._data.sort_values(by=gbcol, inplace=True)
        self._groupingcolumn = gbcol
        self._grouplabels = self._data[gbcol].unique()
        
    def split_to_groups(self, col, inplace=False):
        """
        split selected data into groups

        :param col: name (str) or index (int, 0-indexed, based on self._data) of column to group by
        :param inplace: bool; parameter spcifiying whether drop of grouping column is performed inplace (True) 
                              or a copy is returned (default, False)
        """
        self._check_selection()
        if self._grouplabels is None:
            self.group(col)

        groups = []
        for _, grouplabel in enumerate(self._grouplabels):
            groupdata = self._data.loc[self._data[self._groupingcolumn] == grouplabel]
            if inplace:
                groupdata.drop(columns=self._groupingcolumn,inplace=inplace)
            else:
                groupdata = groupdata.drop(columns=self._groupingcolumn,inplace=inplace)
            groups.append(groupdata)

        self._groups = groups

        self._groupcolmetadata = copy.deepcopy(self._colmetadata)
        for i, d in enumerate(self._colmetadata):
            if d['ColNames'] == self._groupingcolumn:
                self._groupcolmetadata.pop(i)


    def get_joint_discretization(self,method=None,return_result=False,*args,**kwargs):
        """
        Obtain discretization of columns/questions with continuous answer space,
        common across all groups being considered

        :param method: keyword specifying discretization method. See discretization documentation
        :param return_result: bool; default False. Either return obtained discretization, or 
                              update self._colmetadata (default) of PreProcesor instance 
        :param *args: optional arguments to be passed to discretization methods
        :param *kwargs: optional keyword arguments to be passed to discrettization methods
        :return discretization: list of dicts with ColNames key and density_method key of arrays with bin borders
        """

        self._check_selection()
        disc = discretize(self._data,self._colmetadata,method=method,*args,**kwargs)

        if return_result:
            return disc
        else:
            for i, ent in enumerate(disc):
                for j, md in enumerate(self._colmetadata):
                    if ent['ColNames'] == md['ColNames']:
                        self._colmetadata[j].update(ent)
                        break
                    else:
                        pass 
