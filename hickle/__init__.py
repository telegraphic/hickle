# hickle imports
from .__version__ import __version__
from . import hickle
from .hickle import *
from .fileio import ClosedFileError, FileError
from .hickle_bshuf import BitShuffleLz4

# All declaration
__all__ = ['hickle', 'ClosedFileError', 'FileError']
__all__.extend(hickle.__all__)

# Author declaration
__author__ = "Danny Price, Ellert van der Velden and contributors"
