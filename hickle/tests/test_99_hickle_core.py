#! /usr/bin/env python
# encoding: utf-8
"""
# test_hickle.py

Unit tests for hickle module.

"""


# %% IMPORTS
# Built-in imports
from collections import OrderedDict as odict
import os
import re
from pprint import pprint


# Package imports
import pytest
import dill as pickle
import h5py
import numpy as np
from py.path import local

# hickle imports
from hickle import dump, helpers, hickle, load, lookup, fileio

# Set current working directory to the temporary directory
local.get_temproot().chdir()


# %% GLOBALS

# %% HELPER DEFINITIONS

# %% FIXTURES

@pytest.fixture
def h5_data(request):
    """
    create dummy hdf5 test data file for testing PyContainer and H5NodeFilterProxy
    """
    import h5py as h5
    dummy_file = h5.File('hickle_core_{}.hdf5'.format(request.function.__name__),'w')
    filename = dummy_file.filename
    test_data = dummy_file.create_group("root_group")
    yield test_data
    dummy_file.close()
    
@pytest.fixture
def test_file_name(request):
    yield "{}.hkl".format(request.function.__name__)

# %% FUNCTION DEFINITIONS
        
def test_recursive_dump(h5_data,compression_kwargs):
    """
    test _dump function and that it properly calls itself recursively 
    """

    # check that dump function properly creates a list dataset and
    # sets appropriate values for 'type' and 'base_type' attributes
    data = simple_list = [1,2,3,4]
    with lookup.ReferenceManager.create_manager(h5_data) as memo:
        with lookup.LoaderManager.create_manager(h5_data) as loader:
            hickle._dump(data, h5_data, "simple_list",memo,loader,**compression_kwargs)
            dumped_data = h5_data["simple_list"]
            assert memo.resolve_type(dumped_data) == (data.__class__,b'list',False)
            assert np.all(dumped_data[()] == simple_list)
        
            # check that dump function properly creates a group representing
            # a dictionary and its keys and values and sets appropriate values
            # for 'type', 'base_type' and 'key_base_type' attributes
            data = {
                '12':12,
                (1,2,3):'hallo'
            }
            hickle._dump(data, h5_data, "some_dict",memo,loader,**compression_kwargs)
            dumped_data = h5_data["some_dict"]
            assert memo.resolve_type(dumped_data) == (data.__class__,b'dict',True)
        
            # check that the name of the resulting dataset for the first dict item
            # resembles double quoted string key and 'type', 'base_type 'key_base_type'
            # attributes the resulting dataset are set accordingly
            first_item = dumped_data['"12"']
            assert first_item[()] == 12 and first_item.attrs['key_base_type'] in (b'str','str')
            assert memo.resolve_type(first_item) == (data['12'].__class__,b'int',False)
            #assert first_item.attrs['base_type'] == b'int'
            #assert first_item.attrs['type'] == pickle.dumps(data['12'].__class__) 
            
            # check that second item is converted into key value pair group, that
            # the name of that group reads 'data0' and that 'type', 'base_type' and
            # 'key_base_type' attributes are set accordingly
            second_item = dumped_data.get("data0",None)
            if second_item is None:
                second_item = dumped_data["data1"]
            assert second_item.attrs['key_base_type'] in (b'key_value','key_value')
            assert memo.resolve_type(second_item) == (tuple,b'tuple',True)
            #assert second_item.attrs['type'] == pickle.dumps(tuple)
        
            # check that content of key value pair group resembles key and value of
            # second dict item
            key = second_item['data0']
            value = second_item['data1']
            assert np.all(key[()] == (1,2,3))
            # and key.attrs['base_type'] == b'tuple'
            assert memo.resolve_type(key) == (tuple,b'tuple',False)
            assert bytes(value[()]) == 'hallo'.encode('utf8')
            # and value.attrs['base_type'] == b'str'
            assert memo.resolve_type(value) == (str,b'str',False)
        
            # check that objects for which no loader has been registered or for which
            # available loader raises NotHicklable exception are handled by 
            # create_pickled_dataset function 
            def fail_create_dict(py_obj,h_group,name,**kwargs):
                raise helpers.NotHicklable("test loader shrugg")
            loader.types_dict.maps.insert(0,{dict:(fail_create_dict,*loader.types_dict[dict][1:])})
            memo_backup = memo.pop(id(data),None)
            with pytest.warns(lookup.SerializedWarning):
                hickle._dump(data, h5_data, "pickled_dict",memo,loader,**compression_kwargs)
            dumped_data = h5_data["pickled_dict"]
            assert bytes(dumped_data[()]) == pickle.dumps(data)
            loader.types_dict.maps.pop(0)
            memo[id(data)] = memo_backup
    
def test_recursive_load(h5_data,compression_kwargs):
    """
    test _load function and that it properly calls itself recursively 
    """

    # check that simple scalar value is properly restored on load from
    # corresponding dataset
    data = 42
    data_name = "the_answer"
    with lookup.ReferenceManager.create_manager(h5_data) as memo:
        with lookup.LoaderManager.create_manager(h5_data) as loader:
            hickle._dump(data, h5_data, data_name,memo,loader,**compression_kwargs)
            py_container = hickle.RootContainer(h5_data.attrs,b'hickle_root',hickle.RootContainer)
            hickle._load(py_container, data_name, h5_data[data_name],memo,loader)
            assert py_container.convert() == data
        
            # check that dict object is properly restored on load from corresponding group
            data = {'question':None,'answer':42}
            data_name = "not_formulated"
            hickle._dump(data, h5_data, data_name,memo,loader,**compression_kwargs)
            py_container = hickle.RootContainer(h5_data.attrs,b'hickle_root',hickle.RootContainer)
            hickle._load(py_container, data_name, h5_data[data_name],memo,loader)
            assert py_container.convert() == data
        
            
            # check that objects for which no loader has been registered or for which
            # available loader raises NotHicklable exception are properly restored on load
            # from corresponding copy protocol group or pickled data string 
            def fail_create_dict(py_obj,h_group,name,**kwargs):
                raise helpers.NotHicklable("test loader shrugg")
            loader.types_dict.maps.insert(0,{dict:(fail_create_dict,*loader.types_dict[dict][1:])})
            data_name = "pickled_dict"
            memo_backup = memo.pop(id(data),None)
            with pytest.warns(lookup.SerializedWarning):
                hickle._dump(data, h5_data, data_name,memo,loader,**compression_kwargs)
            hickle._load(py_container, data_name, h5_data[data_name],memo,loader)
            assert py_container.convert() == data
            loader.types_dict.maps.pop(0)
            memo[id(data)] = memo_backup

# %% ISSUE RELATED TESTS

def test_invalid_file(compression_kwargs):
    """ Test if trying to use a non-file object fails. """

    with pytest.raises(hickle.FileError):
        dump('test', (),**compression_kwargs)


def test_binary_file(test_file_name,compression_kwargs):
    """ Test if using a binary file works

    https://github.com/telegraphic/hickle/issues/123"""

    filename = test_file_name.replace(".hkl",".hdf5")
    with open(filename, "w") as f:
        with pytest.raises(hickle.FileError):
            hickle.dump(None, f,**compression_kwargs)
    with open(filename, "w+") as f:
        with pytest.raises(hickle.FileError):
            hickle.dump(None, f,**compression_kwargs)

    with open(filename, "wb") as f:
        with pytest.raises(hickle.FileError):
            hickle.dump(None, f,**compression_kwargs)

    with open(filename, "w+b") as f:
        hickle.dump(None, f,**compression_kwargs)


def test_file_open_close(test_file_name,h5_data,compression_kwargs):
    """ https://github.com/telegraphic/hickle/issues/20 """
    import h5py
    f = h5py.File(test_file_name.replace(".hkl",".hdf"), 'w')
    a = np.arange(5)

    dump(a, test_file_name,**compression_kwargs)
    dump(a, test_file_name,**compression_kwargs)

    dump(a, f, mode='w',**compression_kwargs)
    f.close()
    with pytest.raises(hickle.ClosedFileError):
        dump(a, f, mode='w',**compression_kwargs)
    h5_data.create_dataset('nothing',data=[])
    with pytest.raises(ValueError,match = r"Unable\s+to\s+create\s+group\s+\(name\s+already\s+exists\)"):
        dump(a,h5_data.file,path="/root_group",**compression_kwargs)


def test_hdf5_group(test_file_name,compression_kwargs):
    import h5py
    hdf5_filename = test_file_name.replace(".hkl",".hdf5")
    file = h5py.File(hdf5_filename, 'w')
    group = file.create_group('test_group')
    a = np.arange(5)
    dump(a, group,**compression_kwargs)
    file.close()

    a_hkl = load(hdf5_filename, path='/test_group')
    assert np.allclose(a_hkl, a)

    file = h5py.File(hdf5_filename, 'r+')
    group = file.create_group('test_group2')
    b = np.arange(8)

    dump(b, group, path='deeper/and_deeper',**compression_kwargs)
    file.close()

    with pytest.raises(ValueError):
        b_hkl = load(hdf5_filename, path='/test_group2/deeper_/and_deeper')
    b_hkl = load(hdf5_filename, path='/test_group2/deeper/and_deeper')
    assert np.allclose(b_hkl, b)

    file = h5py.File(hdf5_filename, 'r')
    b_hkl2 = load(file['test_group2'], path='deeper/and_deeper')
    assert np.allclose(b_hkl2, b)
    file.close()



def test_with_open_file(test_file_name,compression_kwargs):
    """
    Testing dumping and loading to an open file

    https://github.com/telegraphic/hickle/issues/92"""

    lst = [1]
    tpl = (1,)
    dct = {1: 1}
    arr = np.array([1])

    with h5py.File(test_file_name, 'w') as file:
        dump(lst, file, path='/lst',**compression_kwargs)
        dump(tpl, file, path='/tpl',**compression_kwargs)
        dump(dct, file, path='/dct',**compression_kwargs)
        dump(arr, file, path='/arr',**compression_kwargs)

    with h5py.File(test_file_name, 'r') as file:
        assert load(file, '/lst') == lst
        assert load(file, '/tpl') == tpl
        assert load(file, '/dct') == dct
        assert load(file, '/arr') == arr


def test_load(test_file_name,compression_kwargs):
    a = set([1, 2, 3, 4])
    b = set([5, 6, 7, 8])
    c = set([9, 10, 11, 12])
    z = (a, b, c)
    z = [z, z]
    z = (z, z, z, z, z)

    print("Original:")
    pprint(z)
    dump(z, test_file_name, mode='w',**compression_kwargs)

    print("\nReconstructed:")
    z = load(test_file_name)
    pprint(z)




def test_multi_hickle(test_file_name,compression_kwargs):
    """ Dumping to and loading from the same file several times

    https://github.com/telegraphic/hickle/issues/20"""

    a = {'a': 123, 'b': [1, 2, 4]}

    if os.path.exists(test_file_name):
        os.remove(test_file_name)
    dump(a, test_file_name, path="/test", mode="w",**compression_kwargs)
    dump(a, test_file_name, path="/test2", mode="r+",**compression_kwargs)
    dump(a, test_file_name, path="/test3", mode="r+",**compression_kwargs)
    dump(a, test_file_name, path="/test4", mode="r+",**compression_kwargs)

    load(test_file_name, path="/test")
    load(test_file_name, path="/test2")
    load(test_file_name, path="/test3")
    load(test_file_name, path="/test4")


def test_improper_attrs(test_file_name,compression_kwargs):
    """
    test for proper reporting missing mandatory attributes for the various
    supported file versions
    """

    # check that missing attributes which disallow to identify
    # hickle version are reported
    data = "my name? Ha I'm Nobody"
    dump(data,test_file_name,**compression_kwargs)
    manipulated = h5py.File(test_file_name,"r+")
    root_group = manipulated.get('/')
    root_group.attrs["VERSION"] = root_group.attrs["HICKLE_VERSION"]
    del root_group.attrs["HICKLE_VERSION"]
    manipulated.flush()
    with pytest.raises(
        ValueError,
        match= r"Provided\s+argument\s+'file_obj'\s+does\s+not\s+appear"
               r"\s+to\s+be\s+a\s+valid\s+hickle\s+file!.*"
    ):
        load(manipulated)


# %% MAIN SCRIPT
if __name__ == '__main__':
    """ Some tests and examples """
    from _pytest.fixtures import FixtureRequest
    from hickle.tests.conftest import compression_kwargs

    for h5_root,keywords in (
        ( h5_data(request),compression_kwargs(request) )
        for request in (FixtureRequest(test_recursive_dump),)
    ):
        test_recursive_dump(h5_root,keywords)
    for h5_root,keywords in (
        ( h5_data(request),compression_kwargs(request) )
        for request in (FixtureRequest(test_recursive_load),)
    ):
        test_recursive_load(h5_root,keywords)
    for keywords in compression_kwargs(FixtureRequest(test_recursive_dump)):
        test_invalid_file(keywords)
    for filename,keywords in (
        ( test_file_name(request),compression_kwargs(request) )
        for request in (FixtureRequest(test_binary_file),)
    ):
        test_binary_file(filename,keywords)
    for h5_root,filename,keywords in (
        ( h5_data(request),test_file_name(request),compression_kwargs(request) )
        for request in (FixtureRequest(test_file_open_close),)
    ):
        test_file_open_close(h5_root,filename,keywords)
    for filename,keywords in (
        ( test_file_name(request),compression_kwargs(request) )
        for request in (FixtureRequest(test_hdf5_group),)
    ):
        test_hdf5_group(filename,keywords)
    for filename,keywords in (
        ( test_file_name(request),compression_kwargs(request) )
        for request in (FixtureRequest(test_with_open_file),)
    ):
        test_with_open_file(filename,keywords)
    for filename,keywords in (
        ( test_file_name(request),compression_kwargs(request) )
        for request in (FixtureRequest(test_load),)
    ):
        test_load(filename,keywords)
    for filename,keywords in (
        ( test_file_name(request),compression_kwargs(request) )
        for request in (FixtureRequest(test_multi_hickle),)
    ):
        test_multi_hickle(filename,keywords)
    for filename,keywords in (
        ( test_file_name(request),compression_kwargs(request) )
        for request in (FixtureRequest(test_improper_attrs),)
    ):
        test_improper_attrs(filename,keywords)

