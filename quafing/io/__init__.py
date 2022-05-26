import os

from .excel_handler import ExcelHandler as excel

io_handlers = {
	'.xls': excel,
	'.xlsx': excel,
	'.xlsm': excel,
	'.xlsb': excel,
	'.odf': excel,
	'.ods': excel,
	'.odt': excel
}

def get_io_handler(path, mode, format=None, overwrite=False):
	"""
	Return instance of of specific IOHandler already initialized tto read or write mode.

	:param path: path where IO operation required
	:param mode: 'r' for reading, 'w' for writing
	:param format: format of file to be handled. Attempt to guess based on extension if omitted.
	:param overwrite: if working in write mode, allow overwriting of exisitng file
	:return: instane of the IOHandler
	"""

	if format is None:
		_root, format = os.path.splitext(path)
	format = format.lower()
	_check_format(format)
	io_handler = io_handlers[format]
	return io_handler(path, mode, overwrite=overwrite)

def _check_format(format):
	if format not in io_handlers:
		raise NotImplementedError(
			"File format %s unknown. Supported format are : %s" % (format, ', '.join(io_handlers.keys())))
