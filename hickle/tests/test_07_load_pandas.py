#! /usr/bin/env python
# encoding: utf-8
"""
# test_load_pandas

Unit tests for hickle module -- pandas loader.

"""

# TODO add tests for all loader related dump_fcn, load_fcn functions
#      and PyContainer classes as soon as there exists any pandas
#      specific loader

# %% IMPORTS
# Package imports
import h5py as h5
import numpy as np
import pytest
import pandas as pd
from py.path import local

# hickle imports
import hickle.loaders.load_pandas as load_pandas


# Set the current working directory to the temporary directory
local.get_temproot().chdir()

# %% FIXTURES

@pytest.fixture
def h5_data(request):
    """
    create dummy hdf5 test data file for testing PyContainer and H5NodeFilterProxy
    """
    dummy_file = h5.File('test_load_builtins.hdf5','w')
    dummy_file = h5.File('load_numpy_{}.hdf5'.format(request.function.__name__),'w')
    filename = dummy_file.filename
    test_data = dummy_file.create_group("root_group")
    yield test_data
    dummy_file.close()

# %% FUNCTION DEFINITIONS
def test_nothing_yet_totest(h5_data,compression_kwargs):
    """
    dummy test function to be removed as soon as load_pandas loader module
    contains dump_fcn, load_fcn and PyContainer functions and classes
    for pandas arrays and objects.
    """

# %% MAIN SCRIPT
if __name__ == "__main__":
    from _pytest.fixtures import FixtureRequest
    from conftest import compression_kwargs
    for h5_root,keywords in (
        ( h5_data(request),compression_kwargs(request) )
        for request in (FixtureRequest(test_nothing_yet_totest),)
    ):
        test_nothing_yet_totest(h5_root,keywords)
