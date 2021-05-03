# encoding: utf-8
"""
# load_numpy.py

Utilities and dump / load handlers for handling numpy and scipy arrays

"""

# %% IMPORTS
# Package imports
import numpy as np
import types

# hickle imports
from hickle.loaders.load_builtins import create_listlike_dataset,ListLikeContainer
from hickle.helpers import PyContainer,no_compression


# %% FUNCTION DEFINITIONS

def create_np_scalar_dataset(py_obj, h_group, name, **kwargs):
    """ dumps an numpy.dtype object to h5py file

    Parameters
    ----------
    py_obj (numpy.scalar):
        python object to dump; should be a numpy scalar, e.g.  numpy.float16(1)

    h_group (h5.File.group):
        group to dump data into.

    name (str):
         the name of the resulting dataset

    kwargs (dict):
        keyword arguments to be passed to create_dataset function

    Returns
    -------
    tuple containing h5py.Dataset and empty list of subitems
    """

    d = h_group.create_dataset(name, data=py_obj, **no_compression(kwargs))

    d.attrs["np_dtype"] = py_obj.dtype.str.encode("ascii")
    return d,()


def create_np_dtype(py_obj, h_group, name, **kwargs):
    """ dumps an numpy dtype object to h5py file

    Parameters
    ----------
    py_obj (numpy.dtype):
        python object to dump; should be a numpy dtype, e.g.  numpy.float16

    h_group (h5.File.group):
        group to dump data into.

    name (str):
        the name of the resulting dataset

    kwargs (dict):
        keyword arguments to be passed to create_dataset function

    Returns
    -------
    tuple containing h5py.Dataset and empty list of subitems
    """
    d = h_group.create_dataset(name, data=bytearray(py_obj.str,"ascii"), **kwargs)
    return d,()


def create_np_array_dataset(py_obj, h_group, name, **kwargs):
    """ dumps an ndarray object to h5py file

    Parameters
    ----------
    py_obj (numpy.ndarray):
        python object to dump; should be a numpy.ndarray or numpy.ma.array (masked)

    h_group (h5.File.group):
        group to dump data into.

    name (str):
        the name of the resulting dataset or group

    kwargs (dict):
        keyword arguments to be passed to create_dataset function

    Returns
    -------
    tuple containing h5py.Datset and empty list of subitems or h5py.Group
    and iterable of subitems
    """

    # Obtain dtype of py_obj
    dtype = py_obj.dtype

    # Check if py_obj contains strings
    if "str" in dtype.name:
        if py_obj.ndim < 1:
            # convert string to utf8 encoded bytearray
            string_data = bytearray(py_obj.item(),"utf8") if 'bytes' not in dtype.name else memoryview(py_obj.item())
            string_data = np.array(string_data,copy = False)
            string_data.dtype = 'S1'
            h_node = h_group.create_dataset(name,data = string_data,shape=(1,string_data.size),**kwargs)
            sub_items = ()
        else:
            # store content as list of strings
            h_node,sub_items = create_listlike_dataset(py_obj.tolist(), h_group, name, **kwargs)
    elif dtype.name == 'object':
        # If so, convert py_obj to list
        py_obj = py_obj.tolist()

        # Check if py_obj is a list
        if isinstance(py_obj, list):
            # If so, dump py_obj into the current group
            h_node,sub_items = create_listlike_dataset(py_obj, h_group, name, **kwargs)
        else:
            # If not, create a new group and dump py_obj into that
            h_node = h_group.create_group(name)
            sub_items = ("data",py_obj,{},kwargs),
    else:
        h_node = h_group.create_dataset(
            name, data=py_obj, **( no_compression(kwargs) if "bytes" in dtype.name else kwargs )
        )
        sub_items = ()
    h_node.attrs['np_dtype'] = dtype.str.encode('ascii')
    return h_node,sub_items

def create_np_masked_array_dataset(py_obj, h_group, name, **kwargs):
    """ dumps an numpy.ma.core.MaskedArray object to h5py file

    Parameters
    ----------
    py_obj (numpy.ma.array):
        python object to dump; should be a numpy.ndarray or numpy.ma.array (masked)

    h_group (h5.File.group):
        group to dump data into.

    name (str):
        the name of the resulting dataset or group

    kwargs (dict):
        keyword arguments to be passed to create_dataset function

    Returns
    -------
    tuple containing h5py.Group and subitems list representing masked array contents:

    """

    # Obtain dtype of py_obj

    h_node = h_group.create_group(name)
    h_node.attrs['np_dtype'] = py_obj.dtype.str.encode('ascii')
    return h_node,(("data",py_obj.data,{},kwargs),('mask',py_obj.mask,{},kwargs))


def load_np_dtype_dataset(h_node,base_type,py_obj_type):
    """
    restores dtype from dataset

    Parameters
    ----------
    h_node (h5py.Dataset):
        the hdf5 node to load data from

    base_type (bytes):
        bytes string denoting base_type

    py_obj_type (numpy.dtype):
        final type of restored dtype

    Returns
    -------
    resulting numpy.dtype
    """
    return np.dtype(bytes(h_node[()]))


def load_np_scalar_dataset(h_node,base_type,py_obj_type):
    """
    restores scalar value from dataset

    Parameters
    ----------
    h_node (h5py.Dataset):
        the hdf5 node to load data from

    base_type (bytes):
        bytes string denoting base_type

    py_obj_type (numpy.dtype):
        final type of restored dtype

    Returns
    -------
    resulting numpy.scalar
    """

    dtype = np.dtype(h_node.attrs["np_dtype"])
    return dtype.type(h_node[()])


def load_ndarray_dataset(h_node,base_type,py_obj_type):
    """
    restores ndarray like object from dataset

    Parameters
    ----------
    h_node (h5py.Dataset):
        the hdf5 node to load data from

    base_type (bytes):
        bytes string denoting base_type

    py_obj_type (numpy.ndarray, numpy.ma.array, ...):
        final type of restored array

    Returns
    -------
    resulting numpy.ndarray, numpy.ma.array
    """
    dtype = np.dtype(h_node.attrs['np_dtype'])
    if "str" in dtype.name:
        string_data = h_node[()]
        if h_node.dtype.itemsize <= 1 or 'bytes' not in h_node.dtype.name:
            # in hickle 4.0.X numpy.ndarrays containing multiple strings are 
            # not converted to list of string but saved as ar consequently
            # itemsize of dtype is > 1
            string_data = bytes(string_data).decode("utf8")
        return np.array(string_data,copy=False,dtype=dtype)
    if issubclass(py_obj_type,np.matrix):
        return py_obj_type(data=h_node[()],dtype=dtype)
    # TODO how to restore other ndarray derived object_types
    # simply using classname for casting does not work, in
    # case they use the same interface like numpy.ndarray
    return np.array(h_node[()], dtype=dtype)


def load_ndarray_masked_dataset(h_node,base_type,py_obj_type):
    """
    restores masked array from data and mask datasets as stored by
    hickle version 4.0.0

    Parameters
    ----------
    h_node (h5py.Dataset):
        the hdf5 node to load data from

    base_type (bytes):
        bytes string denoting base_type

    py_obj_type (numpy.ndarray, numpy.ma.array, ...):
        final type of restored array

    Returns
    -------
    resulting numpy.ndarray, numpy.ma.array
    """
    masked_array = NDMaskedArrayContainer(h_node.attrs,base_type,py_obj_type)
    masked_array.append('data',h_node[()],h_node.attrs),
    mask_path = "{}_mask".format(h_node.name)
    h_root = h_node.parent
    h_node_mask = h_root.get(mask_path,None)
    if h_node_mask is None:
        raise ValueError("mask not found")
    masked_array.append('mask',h_node_mask,h_node_mask.attrs)
    return masked_array.convert()

class NDArrayLikeContainer(ListLikeContainer):
    """
    PyContainer used to restore complex ndarray from h5py.Group node
    """

    __slots__ = ()
    
    def append(self,name,item,h5_attrs):

        # if group contains only one item which either has been
        # dumped using create_pickled_dataset or its name reads
        # data than assume single non list-type object otherwise
        # pass item on to append method of ListLikeContainer
        if h5_attrs.get("base_type",'') == b'pickle' or name == "data":
            self._content = item
        else:
            super(NDArrayLikeContainer,self).append(name,item,h5_attrs)
    
    def convert(self):
        data = np.array(self._content,dtype = self._h5_attrs['np_dtype'])
        return data if data.__class__ is self.object_type or isinstance(self.object_type,types.LambdaType) else self.object_type(data)

class NDMaskedArrayContainer(PyContainer):
    """
    PyContainer used to restore masked array stored as dedicated h5py.Group
    """
    __slots__ = ()

    def __init__(self,h5_attrs,base_type,object_type):
        super(NDMaskedArrayContainer,self).__init__(h5_attrs,base_type,object_type,_content = {})

    def append(self,name,item,h5_attrs):
        self._content[name] = item

    def convert(self):
        dtype = self._h5_attrs['np_dtype']
        data = np.ma.array(self._content['data'], mask=self._content['mask'], dtype=dtype)
        return data if data.__class__ is self.object_type or isinstance(self.object_type,types.LambdaType) else self.object_type(data)

#####################
# Lookup dictionary #
#####################

# %% REGISTERS
class_register = [
    [np.dtype, b"np_dtype", create_np_dtype, load_np_dtype_dataset],
    [np.number, b"np_scalar", create_np_scalar_dataset, load_np_scalar_dataset,None,False],

    # for all scalars which are not derived from numpy.number which itself is numpy.generic subclass
    # to properly catch and handle they will be caught by the following
    [np.generic, b"np_scalar", create_np_scalar_dataset, load_np_scalar_dataset,None,False],

    [np.ndarray, b"ndarray", create_np_array_dataset, load_ndarray_dataset,NDArrayLikeContainer],
    [np.ma.core.MaskedArray, b"ndarray_masked", create_np_masked_array_dataset, None,NDMaskedArrayContainer],

    # NOTE: The following is load only
    #       just needed to link old ndarray_masked_data base_type to load_ndarray_masked_dataset
    #       loader module selection will be triggered by numpy.ma.core.MaskedArray object_type anyway
    #       but base_type is used to select proper load_function
    [np.ma.core.MaskedArray, b"ndarray_masked_data",None , load_ndarray_masked_dataset,None,False,'hickle-4.x'],

    # NOTE: numpy.matrix is obsolete and just an alias for numpy.ndarray therefore 
    # to keep things simple numpy.matrix will be handled by same functions as 
    # numpy.ndarray. As long as just required cause numpy.ma.core.MaskedArray 
    # uses it for data
    [np.matrix, b"np_matrix", create_np_array_dataset, load_ndarray_dataset]
]

exclude_register = [
    (b"ndarray_masked_mask",'hickle-4.x')
]
