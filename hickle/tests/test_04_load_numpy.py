#! /usr/bin/env python
# encoding: utf-8
"""
# test_load_numpy

Unit tests for hickle module -- numpy loader.

"""
import pytest

import sys

# %% IMPORTS
# Package imports
import h5py as h5
import numpy as np
import hickle.loaders.load_numpy as load_numpy
from py.path import local


# Set current working directory to the temporary directory
local.get_temproot().chdir()

# %% GLOBALS

NESTED_DICT = {
    "level1_1": {
        "level2_1": [1, 2, 3],
        "level2_2": [4, 5, 6]
    },
    "level1_2": {
        "level2_1": [1, 2, 3],
        "level2_2": [4, 5, 6]
    },
    "level1_3": {
        "level2_1": {
            "level3_1": [1, 2, 3],
            "level3_2": [4, 5, 6]
        },
        "level2_2": [4, 5, 6]
    }
}

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

def test_create_np_scalar(h5_data,compression_kwargs):
    """
    tests proper storage and loading of numpy scalars
    """

    # check that scalar dataset is created for numpy scalar
    scalar_data = np.float64(np.pi)
    dtype = scalar_data.dtype
    h_dataset,subitems = load_numpy.create_np_scalar_dataset(scalar_data,h5_data,"scalar_data",**compression_kwargs)
    assert isinstance(h_dataset,h5.Dataset) and iter(subitems) and not subitems
    assert h_dataset.attrs['np_dtype'] in ( dtype.str.encode('ascii'),dtype.str)
    assert  h_dataset[()] == scalar_data
    assert load_numpy.load_np_scalar_dataset(h_dataset,b'np_scalar',scalar_data.__class__) == scalar_data

    # check that numpy.bool_ scarlar is properly stored and reloaded
    scalar_data = np.bool_(True)
    dtype = scalar_data.dtype
    h_dataset,subitems = load_numpy.create_np_scalar_dataset(scalar_data,h5_data,"generic_data",**compression_kwargs)
    assert isinstance(h_dataset,h5.Dataset) and iter(subitems) and not subitems
    assert h_dataset.attrs['np_dtype'] in ( dtype.str.encode('ascii'),dtype.str) and h_dataset[()] == scalar_data
    assert load_numpy.load_np_scalar_dataset(h_dataset,b'np_scalar',scalar_data.__class__) == scalar_data

def test_create_np_dtype(h5_data,compression_kwargs):
    """
    test proper creation and loading of dataset representing numpy dtype
    """ 
    dtype = np.dtype(np.int16)
    h_dataset,subitems = load_numpy.create_np_dtype(dtype, h5_data,"dtype_string",**compression_kwargs)
    assert isinstance(h_dataset,h5.Dataset) and iter(subitems) and not subitems
    assert bytes(h_dataset[()]).decode('ascii') == dtype.str
    assert load_numpy.load_np_dtype_dataset(h_dataset,'np_dtype',np.dtype) == dtype

def test_create_np_ndarray(h5_data,compression_kwargs):
    """
    test proper creation and loading of numpy ndarray
    """

    # check that numpy array representing python utf8 string is properly 
    # stored as bytearray dataset and reloaded from
    np_array_data = np.array("im python string")
    h_dataset,subitems = load_numpy.create_np_array_dataset(np_array_data,h5_data,"numpy_string_array",**compression_kwargs)
    assert isinstance(h_dataset,h5.Dataset) and iter(subitems) and not subitems
    assert bytes(h_dataset[()]) == np_array_data.tolist().encode("utf8")
    assert h_dataset.attrs["np_dtype"] in ( np_array_data.dtype.str.encode("ascii"),np_array_data.dtype.str)
    assert load_numpy.load_ndarray_dataset(h_dataset,b'ndarray',np.ndarray) == np_array_data

    # check that numpy array representing python bytes string is properly
    # stored as bytearray dataset and reloaded from
    np_array_data = np.array(b"im python bytes")
    h_dataset,subitems = load_numpy.create_np_array_dataset(np_array_data,h5_data,"numpy_bytes_array",**compression_kwargs)
    assert isinstance(h_dataset,h5.Dataset) and iter(subitems) and not subitems
    assert h_dataset[()] == np_array_data.tolist()
    assert h_dataset.attrs["np_dtype"] in ( np_array_data.dtype.str.encode("ascii"),np_array_data.dtype.str)
    assert load_numpy.load_ndarray_dataset(h_dataset,b'ndarray',np.ndarray) == np_array_data

    # check that numpy array with dtype object representing list of various kinds
    # of objects is converted to list before storing and reloaded properly from this
    # list representation

    # NOTE: simplified as mixing items of varying length receives
    # VisibleDeprecationWarning from newer numpy versions
    #np_array_data = np.array([[NESTED_DICT], ('What is this?',), {1, 2, 3, 7, 1}])
    np_array_data = np.array([NESTED_DICT])#, ('What is this?',), {1, 2, 3, 7, 1}])
    h_dataset,subitems = load_numpy.create_np_array_dataset(np_array_data,h5_data,"numpy_list_object_array",**compression_kwargs)
    ndarray_container = load_numpy.NDArrayLikeContainer(h_dataset.attrs,b'ndarray',np_array_data.__class__)
    assert isinstance(h_dataset,h5.Group) and iter(subitems)
    assert h_dataset.attrs["np_dtype"] in ( np_array_data.dtype.str.encode("ascii"),np_array_data.dtype.str)
    for index,(name,item,attrs,kwargs) in enumerate(subitems):
        assert name == "data{:d}".format(index) and attrs.get("item_index",None) == index
        assert isinstance(kwargs,dict) and np_array_data[index] == item
        ndarray_container.append(name,item,attrs)
    assert np.all(ndarray_container.convert() == np_array_data)

    # check that numpy array containing multiple strings of length > 1
    # is properly converted to list of strings and restored from its list
    # representation
    np_array_data = np.array(["1313e", "was", "maybe?", "here"])
    h_dataset,subitems = load_numpy.create_np_array_dataset(np_array_data,h5_data,"numpy_list_of_strings_array",**compression_kwargs)
    ndarray_container = load_numpy.NDArrayLikeContainer(h_dataset.attrs,b'ndarray',np_array_data.__class__)
    assert isinstance(h_dataset,h5.Group) and iter(subitems)
    assert h_dataset.attrs["np_dtype"] in ( np_array_data.dtype.str.encode("ascii"),np_array_data.dtype.str)
    for index,(name,item,attrs,kwargs) in enumerate(subitems):
        assert name == "data{:d}".format(index) and attrs.get("item_index",None) == index
        assert isinstance(kwargs,dict) and np_array_data[index] == item
        ndarray_container.append(name,item,attrs)
    assert np.all(ndarray_container.convert() == np_array_data)

    # check that numpy array with object dtype which is converted to single object
    # by ndarray.tolist method is properly stored according to type of object and
    # restored from this representation accordingly
    np_array_data = np.array(NESTED_DICT)
    h_dataset,subitems = load_numpy.create_np_array_dataset(np_array_data,h5_data,"numpy_object_array",**compression_kwargs)
    ndarray_container = load_numpy.NDArrayLikeContainer(h_dataset.attrs,b'ndarray',np_array_data.__class__)
    ndarray_pickle_container = load_numpy.NDArrayLikeContainer(h_dataset.attrs,b'ndarray',np_array_data.__class__)
    assert isinstance(h_dataset,h5.Group) and iter(subitems)
    assert h_dataset.attrs["np_dtype"] in ( np_array_data.dtype.str.encode("ascii"),np_array_data.dtype.str)
    data_set = False
    for name,item,attrs,kwargs in subitems:
        if name == "data":
            assert not data_set and not attrs and isinstance(kwargs,dict)
            assert np_array_data[()] == item
            data_set = True
            ndarray_container.append(name,item,attrs)
            attrs = dict(attrs)
            attrs["base_type"] = b'pickle'
            ndarray_pickle_container.append(name,item,attrs)
        else:
            raise AssertionError("expected single data object")
    assert np.all(ndarray_container.convert() == np_array_data)
    assert np.all(ndarray_pickle_container.convert() == np_array_data)

    # check that numpy.matrix type object is properly stored and reloaded from
    # hickle file.
    # NOTE/TODO: current versions of numpy issue PendingDeprecationWarning when using
    # numpy.matrix. In order to indicate to pytest that this is known and can safely
    # be ignored the warning is captured here. Shall it be that future numpy versions
    # convert PendingDeprecationWarning into any kind of exception like TypeError
    # AttributeError, RuntimeError or alike that also capture these Exceptions not
    # just PendingDeprecationWarning
    with pytest.warns(PendingDeprecationWarning):
        np_array_data = np.matrix([[1, 2], [3, 4]])
        h_dataset,subitems = load_numpy.create_np_array_dataset(np_array_data,h5_data,"numpy_matrix",**compression_kwargs)
        assert isinstance(h_dataset,h5.Dataset) and iter(subitems) and not subitems
        assert np.all(h_dataset[()] == np_array_data)
        assert h_dataset.attrs["np_dtype"] in ( np_array_data.dtype.str.encode("ascii"),np_array_data.dtype.str)
        np_loaded_array_data = load_numpy.load_ndarray_dataset(h_dataset,b'npmatrix',np.matrix)
        assert np.all(np_loaded_array_data == np_array_data)
        assert isinstance(np_loaded_array_data,np.matrix)
        assert np_loaded_array_data.shape == np_array_data.shape
    
def test_create_np_masked_array(h5_data,compression_kwargs):
    """
    test proper creation and loading of numpy.masked arrays
    """

    # check that simple masked array is properly stored and loaded
    masked_array = np.ma.array([1, 2, 3, 4], dtype='float32', mask=[0, 1, 0, 0])
    h_datagroup,subitems = load_numpy.create_np_masked_array_dataset(masked_array, h5_data, "masked_array",**compression_kwargs)
    masked_array_container = load_numpy.NDMaskedArrayContainer(h_datagroup.attrs,b'ndarray_masked',np.ma.array)
    assert isinstance(h_datagroup,h5.Group) and iter(subitems)
    assert h_datagroup.attrs["np_dtype"] in ( masked_array.dtype.str.encode("ascii"),masked_array.dtype.str)
    data_set = mask_set = False
    for name,item,attrs,kwargs in subitems:
        assert isinstance(attrs,dict) and isinstance(kwargs,dict)
        if name == "data":
            assert not data_set and not attrs and np.all(masked_array.data == item) and item is not masked_array
            masked_array_container.append(name,item,attrs)
            data_set = True
        elif name == "mask":
            assert not mask_set and not attrs and np.all(masked_array.mask == item) and item is not masked_array
            masked_array_container.append(name,item,attrs)
            mask_set = True
        else:
            raise AssertionError("expected one data and one mask object")
    assert np.all(masked_array_container.convert() == masked_array)

    # check that format used by hickle version 4.0.0 to encode is properly recognized
    # on loading and masked array is restored accordingly
    h_dataset = h5_data.create_dataset("masked_array_dataset",data = masked_array.data)
    h_dataset.attrs["np_dtype"] = masked_array.dtype.str.encode("ascii")
    with pytest.raises(ValueError,match = r"mask\s+not\s+found"):
        loaded_masked_array = load_numpy.load_ndarray_masked_dataset(h_dataset,b'masked_array_data',np.ma.array)
    h_mask_dataset = h5_data.create_dataset("masked_array_dataset_mask",data = masked_array.mask)
    loaded_masked_array = load_numpy.load_ndarray_masked_dataset(h_dataset,b'masked_array_data',np.ma.array)
    assert np.all(loaded_masked_array == masked_array )
    
# %% MAIN SCRIPT
if __name__ == "__main__":
    from _pytest.fixtures import FixtureRequest
    from hickle.tests.conftest import compression_kwargs
    for h5_root,keywords in (
        ( h5_data(request),compression_kwargs(request) )
        for request in (FixtureRequest(test_create_np_scalar),)
    ):
        test_create_np_scalar(h5_root,keywords)
    for h5_root,keywords in (
        ( h5_data(request),compression_kwargs(request) )
        for request in (FixtureRequest(test_create_np_dtype),)
    ):
        test_create_np_dtype(h5_root,keywords)
    for h5_root,keywords in (
        ( h5_data(request),compression_kwargs(request) )
        for request in (FixtureRequest(test_create_np_ndarray),)
    ):
        test_create_np_ndarray(h5_root,keywords)
    for h5_root,keywords in (
        ( h5_data(request),compression_kwargs(request) )
        for request in (FixtureRequest(test_create_np_masked_array),)
    ):
        test_create_np_masked_array(h5_root,keywords)

    
