# encoding: utf-8
"""
# load_python.py

Handlers for dumping and loading built-in python types.
NB: As these are for built-in types, they are critical to the functioning of hickle.

"""

import h5py as h5

def create_listlike_dataset(py_obj, h_group, call_id=0, **kwargs):
    """ Dumper for list, set, tuple

    Args:
        py_obj: python object to dump; should be list-like
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the iterable.
    """
    dtype = str(type(py_obj))
    obj = list(py_obj)
    d = h_group.create_dataset('data_%i' % call_id, data=obj, **kwargs)
    d.attrs["type"] = [dtype]


def create_python_dtype_dataset(py_obj, h_group, call_id=0, **kwargs):
    """ dumps a python dtype object to h5py file

    Args:
        py_obj: python object to dump; should be a python type (int, float, bool etc)
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the iterable.
    """
    # kwarg compression etc does not work on scalars
    d = h_group.create_dataset('data_%i' % call_id, data=py_obj,
                               dtype=type(py_obj))     #, **kwargs)
    d.attrs["type"] = ['python_dtype']
    d.attrs['python_subdtype'] = str(type(py_obj))


def create_stringlike_dataset(py_obj, h_group, call_id=0, **kwargs):
    """ dumps a list object to h5py file

    Args:
        py_obj: python object to dump; should be string-like (unicode or string)
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the iterable.
    """
    if isinstance(py_obj, str):
        d = h_group.create_dataset('data_%i' % call_id, data=[py_obj], **kwargs)
        d.attrs["type"] = ['string']
    else:
        dt = h5.special_dtype(vlen=unicode)
        dset = h_group.create_dataset('data_%i' % call_id, shape=(1, ), dtype=dt, **kwargs)
        dset[0] = py_obj
        dset.attrs['type'] = ['unicode']


def create_none_dataset(py_obj, h_group, call_id=0, **kwargs):
    """ Dump None type to file

    Args:
        py_obj: python object to dump; must be None object
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the iterable.
    """
    d = h_group.create_dataset('data_%i' % call_id, data=[0], **kwargs)
    d.attrs["type"] = ['none']