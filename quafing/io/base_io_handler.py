""" Abstract IO handler """
import os

class IOHandler(object):
    """
    Abstract IO handler class

    Readers should handle row structured data sets
    Writers should handle tabular data or embedddings
    """
    path =  None

    def __init__(self, path, mode, overwrite=False):
        """
        Perform checks on path

        :param path: path for IO operation
        :param mode: 'r' for reading, 'w' for writing
        :param overwrite: overwrite existing file if it already exists     
	    """

        self.path = path
        if mode == 'r':
            if not os.path.exists(path):
                raise FileNotFoundError('{} not found.'.format(path))

        elif mode == 'w':
            path_directory = os.path.dirname(path)
            if path_directory and not os.path.exists(path_directory):
                raise FileNotFoundError('Output file path does not exist! --> {}'.format(path_directory))
            if os.path.exists(path):
                if not overwrite:
                    # Raise most specific subclass of FileExistsError (3.6) and IOError (2.7).
                    raise FileExistsError('Output file already exists! --> {}'.format(path))
                else:
                    os.remove(path)

    def read(self):
        """
        Read data from specified path

        :return data:
        """

        raise NotImplementedError(
            "Class %s doesn't implement read()"% self.__class__.__name__)

    def write(self, data):
        """
        Write data to disk

        :param data:
        """

        raise NotImplementedError(
    	    "Class %s doesn't implement write()"% self.__class__.__name__)    
