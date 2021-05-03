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
import numpy as np
import h5py

# hickle imports
from hickle.helpers import (
    PyContainer,H5NodeFilterProxy,no_compression,convert_str_attr,convert_str_list_attr
)
from hickle.fileio import FileError,ClosedFileError,file_opener,not_io_base_like
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

    # create file and create a dataset the attributes of which will later on be
    # modified
    import h5py as h5
    dummy_file = h5.File('hickle_helpers_{}.hdf5'.format(request.function.__name__),'w')
    filename = dummy_file.filename
    test_data = dummy_file.create_dataset("somedata",data=dummy_data,dtype='i')
    test_data.attrs['type'] = np.array(pickle.dumps(tuple))
    test_data.attrs['base_type'] = b'tuple'
    test_data.attrs['someattr'] = 12
    test_data.attrs['someother'] = 11

    # write out the file reopen it read only
    dummy_file.flush()
    dummy_file.close()
    dummy_file = h5.File(filename,'r')

    # provide the file and close afterwards
    yield dummy_file
    dummy_file.close()
    
@pytest.fixture
def test_file_name(request):
    yield "{}.hkl".format(request.function.__name__)
    
# %% FUNCTION DEFINITIONS

def test_no_compression():
    """
    test no_compression filter for temporarily hiding compression related
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
    # simply shall yield from passed in iterator
    assert [ item for item in dummy_data ] == list(dummy_data)
    assert dict(container.filter(h5_data)) == {'somedata':h5_data['somedata']}


def test_H5NodeFilterProxy(h5_data):
    """
    tests H5NodeFilterProxy class. This class allows to temporarily rewrite
    attributes of h5py.Group and h5py.Dataset nodes before being loaded by 
    hickle._load method. 
    """

    # load data and try to directly modify 'type' and 'base_type' Attributes
    # which will fail cause hdf5 file is opened for read only
    h5_node = h5_data['somedata']
    with pytest.raises(OSError):
        try:
            h5_node.attrs['type'] = pickle.dumps(list)
        except RuntimeError as re:
            raise OSError(re).with_traceback(re.__traceback__)
    with pytest.raises(OSError):
        try:
            h5_node.attrs['base_type'] = b'list'
        except RuntimeError as re:
            raise OSError(re).with_traceback(re.__traceback__)

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

def test_not_io_base_like(test_file_name):
    """
    test not_io_base_like function for creating replacement methods
    for IOBase.isreadable, IOBase.isseekable and IOBase.writeable
    """
    with open(test_file_name,'w') as f:
        assert not not_io_base_like(f)()
        assert not not_io_base_like(f,'strange_read',0)()
        assert not not_io_base_like(f,'seek',0,'strange_tell')()
    with open(test_file_name,'r') as f:
        assert not_io_base_like(f,('seek',0),('tell',))()
        assert not not_io_base_like(f,('seek',0),('tell',()))()
        assert not_io_base_like(f,('read',0))()
        assert not not_io_base_like(f,('tell',()))()
        assert not_io_base_like(f,('tell',))()
    

def test_file_opener(h5_data,test_file_name):
    """
    test file opener function
    """

    # check that file like object is properly initialized for writing
    filename = test_file_name.replace(".hkl","_{}.{}")
    with open(filename.format("w","hdf5"),"w") as f:
        with pytest.raises(FileError):
            h5_file,path,close_flag = file_opener(f,"root","w",filename="filename")
    with open(filename.format("w","hdf5"),"w+b") as f:
        h5_file,path,close_flag = file_opener(f,"root","w+")
        assert isinstance(h5_file,h5py.File) and path == "/root" and h5_file.mode == 'r+'
        h5_file.close()

    # check that file like object is properly initialized for reading
    with open(filename.format("w","hdf5"),"rb") as f:
        h5_file,path,close_flag = file_opener(f,"root","r")
        assert isinstance(h5_file,h5py.File) and path == "/root" and h5_file.mode == 'r'
        assert close_flag
        h5_file.close()
        # check that only str are accepted as filenames
        with pytest.raises(ValueError):
            h5_file,path,close_flag = file_opener(f,"root","r",filename=12)
        # check that tuple specifying file object and filename string is accepted
        h5_file,path,close_flag = file_opener((f,"not me"),"root","r")
        assert isinstance(h5_file,h5py.File) and path == "/root" and h5_file.mode == 'r'
        assert close_flag
        h5_file.close()
        # check that dict specifying file object and filename is accepted
        h5_file,path,close_flag = file_opener({"file":f,"name":"not me"},"root","r")
        assert isinstance(h5_file,h5py.File) and path == "/root" and h5_file.mode == 'r'
        assert close_flag
        h5_file.close()
        # check that file is rejected if mode used to open and mode passed to file
        # opener do not match
        with pytest.raises(FileError):
            h5_file,path,close_flag = file_opener({"file":f,"name":"not me"},"root","r+")
        with pytest.raises(FileError):
            h5_file,path,close_flag = file_opener({"file":f,"name":"not me"},"root","w")
        with pytest.raises(ValueError):
            h5_file,path,close_flag = file_opener({"file":f,"name":"not me"},"root","+")
    # check that only binary files opened for reading and writing are accepted with
    # mode w
    with open(filename.format("w","hdf5"),"w") as f:
        with pytest.raises(FileError):
            h5_file,path,close_flag = file_opener({"file":f,"name":"not me"},"root","w")

    # check that closed file objects are rejected
    with pytest.raises(ClosedFileError):
        h5_file,path,close_flag = file_opener(f,"root","r")
        

    # check that h5py.File object is properly initialised for writing
    with pytest.raises(FileError):
        h5_file,path,close_flag = file_opener(h5_data,"","w")
    with h5py.File(filename.format("w","whkl"),"w") as hdf5_file:
        h5_file,path,close_flag = file_opener(hdf5_file,"","w")
        assert isinstance(h5_file,h5py.File) and path == "/"
        assert h5_file.mode == 'r+' and not close_flag
        hdf5_group = hdf5_file.create_group("some_group")
    with pytest.raises(ClosedFileError):
        h5_file,path,close_flag = file_opener(hdf5_file,"","w")
    with h5py.File(filename.format("w","whkl"),"r") as hdf5_file:
        h5_file,path,close_flag = file_opener(hdf5_file["some_group"],'',"r")
        assert isinstance(h5_file,h5py.File) and path == "/some_group"
        assert h5_file.mode == 'r' and not close_flag
        

    # check that a new file is created for provided filename and properly initialized
    h5_file,path,close_flag = file_opener(filename.format("w",".hkl"),"root_group","w")
    assert isinstance(h5_file,h5py.File) and path == "/root_group"
    assert h5_file.mode == 'r+' and close_flag
    h5_file.close()
    
    # check that any other object not being a file like object, a h5py.File object or
    # a filename string triggers an  FileError exception
    with pytest.raises(FileError):
        h5_file,path,close_flag = file_opener(object(),"root_group","w")

def test_str_attr_converter():
    """
    test attribute decoder helper functions used to mimic
    h5py >= 3.x behaviour when h5py 2.10 is installed
    """
    ascii_str_val = 'some ascii encoded string attr'
    utf8_str_val = 'some utf8 encoded string attr'
    some_attrs = dict(
        some_attr_ascii = ascii_str_val.encode('ascii'),
        some_attr_utf8 = utf8_str_val.encode('utf8'),
        some_attr_list_ascii = [ strval.encode('ascii') for strval in ascii_str_val.split(' ') ],
        some_attr_list_utf8 = [ strval.encode('utf8') for strval in utf8_str_val.split(' ') ]
    )
    assert convert_str_attr(some_attrs,'some_attr_ascii',encoding='ascii') == ascii_str_val
    assert convert_str_attr(some_attrs,'some_attr_utf8') == utf8_str_val
    assert " ".join(convert_str_list_attr(some_attrs,'some_attr_list_ascii',encoding='ascii')) == ascii_str_val
    assert " ".join(convert_str_list_attr(some_attrs,'some_attr_list_utf8')) == utf8_str_val
    

# %% MAIN SCRIPT
if __name__ == "__main__":
    from _pytest.fixtures import FixtureRequest

    test_no_compression()
    for data in h5_data(FixtureRequest(test_py_container)):
        test_py_container(data)
    for data in h5_data(FixtureRequest(test_py_container)):
        test_H5NodeFilterProxy(data)
    for filename in (
        ( test_file_name(request), )
        for request in (FixtureRequest(test_not_io_base_like),)
    ):
        test_not_io_base_like(filename)
    for h5_root,filename in (
        ( h5_data(request),test_file_name(request) )
        for request in (FixtureRequest(test_file_opener),)
    ):
        test_file_opener(h5_root,filename)
    test_str_attr_converter()

