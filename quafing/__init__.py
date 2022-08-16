"""Documentation about quafing"""
import logging

from quafing.io.load import load
from quafing.preprocessing import PreProcessor
from quafing.multipdf.multipdf import create_mdpdf_collection
from quafing.embedding.embed import get_embedding
from quafing.embedding.embed import get_embedder
from quafing.embedding.embed import plot_embedding

logging.getLogger(__name__).addHandler(logging.NullHandler())

__author__ = "Meiert Willem Grootes"
__email__ = "m.grootes@esciencecenter.nl"
__version__ = "0.1.0-alpha"
