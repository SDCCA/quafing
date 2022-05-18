from quafing.io import get_io_handler

def export(data, path, format=None, overwrite=False, *args, **kwargs):
    """
    write data to a file.

    :param data:
    :param path:
    :param format: file format
    :param overwrite: if path exists, overwrite
    :param args: optional non-keyword arguments to be passed to the format specific writer
	:param kwargs: optional keyword arguments to be passed to the format specific writer
	"""
	writer = get_io_handler(path, mode='w', format=format, overwrite=overwrite)
	writer.write(data, *args, **kwargs)
	 