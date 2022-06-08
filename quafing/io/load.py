from quafing.io import get_io_handler

def load(path, format=None, *args, **kwargs):
	"""
	Read data file

	:param path:
	:param format:
	:param args: optional non-keyword arguments to be passed to the format specific reader
	:param kwargs: optional keyword arguments to be passed to the format speicfic reader
	:return: data
	"""
	reader = get_io_handler(path, mode='r', format=format)
	metadata, data = reader.read(*args, **kwargs)
	return metadata, data
