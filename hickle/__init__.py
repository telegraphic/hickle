from __future__ import absolute_import
from .hickle import *


from pkg_resources import get_distribution, DistributionNotFound
try:
    __version__ = get_distribution('hickle').version
except DistributionNotFound:
    __version__ = '0.0.0 - please install via pip/setup.py'
