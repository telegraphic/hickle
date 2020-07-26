#! /usr/bin/env python
# encoding: utf-8
"""
# test_hickle_helpers.py

Unit tests for hickle module -- helper functions.

"""

import pytest

# %% IMPORTS
# Package imports
import numpy as np
import dill as pickle
import operator

# hickle imports
from hickle.helpers import PyContainer,H5NodeFilterProxy,no_compression
from py.path import local

# Set current working directory to the temporary directory
local.get_temproot().chdir()


# %% DATA DEFINITIONS

dummy_data = (1,2,3)


# %% FIXTURES

@pytest.fixture
def h5_data(request):
    """
    create dummy hdf5 test data file for testing PyContainer and H5NodeFilterProxy
    """

    # create file and create a dataset the attributes of which will lateron be
    # modified
    import h5py as h5
    dummy_file = h5.File('hickle_helpers_{}.hdf5'.format(request.function.__name__),'w')
    filename = dummy_file.filename
    test_data = dummy_file.create_dataset("somedata",data=dummy_data,dtype='i')
    test_data.attrs['type'] = np.array(pickle.dumps(tuple))
    test_data.attrs['base_type'] = b'tuple'
    test_data.attrs['someattr'] = 12
    test_data.attrs['someother'] = 11

    # writeout the file reopen it read only
    dummy_file.flush()
    dummy_file.close()
    dummy_file = h5.File(filename,'r')

    # provide the file and close afterwards
    yield dummy_file
    dummy_file.close()
    
# %% FUNCTION DEFINITIONS

def test_no_compression():
    """
    test no_compression filter for temporarily hiding comression related
    kwargs from h5py.create_dataset method
    """

    # simulate kwargs without compression related
    kwargs = {'hello':1,'word':2}
    assert dict(no_compression(kwargs)) == kwargs
    
    # simulate kwargs including all relevant keyword arguments
    kwargs2 = dict(kwargs)
    kwargs2.update({
        "compression":True,
        "shuffle":True,
        "compression_opts":8,
        "chunks":512,
        "fletcher32":True,
        "scaleoffset":20
    })
    assert dict(no_compression(kwargs2)) == kwargs
    
def test_py_container(h5_data):
    """
    test abstract PyContainer base class defining container interface
    and providing default implementations for append and filter
    """

    # test default implementation of append
    container = PyContainer({},b'list',list)
    container.append('data0',1,{})
    container.append('data1','b',{})

    # ensure that default implementation of convert enforces overload by
    # derived PyContainer classes by raising NotImplementedError
    with pytest.raises(NotImplementedError):
        my_list = container.convert()

    # test default implementation of PyContainer.filter method which
    # simply shall yield from passed in itrator
    assert [ item for item in dummy_data ] == list(dummy_data)
    assert dict(container.filter(h5_data.items())) == {'somedata':h5_data['somedata']}


def test_H5NodeFilterProxy(h5_data):
    """
    tests H5NodeFilterProxy class. This class allows to temporarily rewrite
    attributes of h5py.Group and h5py.Dataset nodes before beeing loaded by 
    hickle._load method. 
    """

    # load data and try to directly modify 'type' and 'base_type' Attributes
    # which will fail cause hdf5 file is opened for read only
    h5_node = h5_data['somedata']
    with pytest.raises(OSError):
        h5_node.attrs['type'] = pickle.dumps(list)
    with pytest.raises(OSError):
        h5_node.attrs['base_type'] = b'list'

    # verify that 'type' expands to tuple before running
    # the remaining tests
    object_type = pickle.loads(h5_node.attrs['type'])
    assert object_type is tuple
    assert object_type(h5_node[()].tolist()) == dummy_data

    # Wrap node by H5NodeFilterProxy and rerun the above tests
    # again. This time modifying Attributes shall be possible.
    h5_node = H5NodeFilterProxy(h5_node)
    h5_node.attrs['type'] = pickle.dumps(list)
    h5_node.attrs['base_type'] = b'list'
    object_type = pickle.loads(h5_node.attrs['type'])
    assert object_type is list

    # test proper pass through of item and attribute access 
    # to wrapped h5py.Group or h5py.Dataset object respective
    assert object_type(h5_node[()].tolist()) == list(dummy_data)
    assert h5_node.shape == np.array(dummy_data).shape
    with pytest.raises(AttributeError,match = r"can't\s+set\s+attribute"):
        h5_node.dtype = np.float32

# %% MAIN SCRIPT
if __name__ == "__main__":
    from _pytest.fixtures import FixtureRequest

    test_no_compression()
    for data in h5_data(FixtureRequest(test_py_container)):
        test_py_container(data)
    for data in h5_data(FixtureRequest(test_py_container)):
        test_H5NodeFilterProxy(data)

