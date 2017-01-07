# encoding: utf-8
"""
# load_numpy.py

Utilities and dump / load handlers for handling numpy and scipy arrays

"""

import numpy as np
import scipy
from scipy import sparse

def check_is_numpy_array(py_obj):
    """ Check if a python object is a numpy array (masked or regular)

    Args:
        py_obj: python object to check whether it is a numpy array

    Returns
        is_numpy (bool): Returns True if it is a numpy array, else False if it isn't
    """

    is_numpy = type(py_obj) in (type(np.array([1])), type(np.ma.array([1])))

    return is_numpy


def check_is_scipy_sparse_array(py_obj):
    """ Check if a python object is a scipy sparse array

    Args:
        py_obj: python object to check whether it is a sparse array

    Returns
        is_numpy (bool): Returns True if it is a sparse array, else False if it isn't
    """
    t_csr = type(scipy.sparse.csr_matrix([0]))
    t_csc = type(scipy.sparse.csc_matrix([0]))
    t_bsr = type(scipy.sparse.bsr_matrix([0]))
    is_sparse = type(py_obj) in (t_csr, t_csc, t_bsr)

    return is_sparse


def create_np_scalar_dataset(py_obj, h_group, call_id=0, **kwargs):
    """ dumps an np dtype object to h5py file

    Args:
        py_obj: python object to dump; should be a numpy scalar, e.g. np.float16(1)
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the iterable.
    """

    # DO NOT PASS KWARGS TO SCALAR DATASETS!
    d = h_group.create_dataset('data_%i' % call_id, data=py_obj)  # **kwargs)
    d.attrs["type"] = ['np_scalar']
    d.attrs["np_dtype"] = str(d.dtype)


def create_np_dtype(py_obj, h_group, call_id=0, **kwargs):
    """ dumps an np dtype object to h5py file

    Args:
        py_obj: python object to dump; should be a numpy scalar, e.g. np.float16(1)
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the iterable.
    """
    d = h_group.create_dataset('data_%i' % call_id, data=[str(py_obj)])
    d.attrs["type"] = ['np_dtype']


def create_np_array_dataset(py_obj, h_group, call_id=0, **kwargs):
    """ dumps an ndarray object to h5py file

    Args:
        py_obj: python object to dump; should be a numpy array or np.ma.array (masked)
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the iterable.
    """
    if isinstance(py_obj, type(np.ma.array([1]))):
        d = h_group.create_dataset('data_%i' % call_id, data=py_obj, **kwargs)
        #m = h_group.create_dataset('mask_%i' % call_id, data=py_obj.mask, **kwargs)
        m = h_group.create_dataset('data_%i_mask' % call_id, data=py_obj.mask, **kwargs)
        d.attrs["type"] = ['ndarray_masked_data']
        m.attrs["type"] = ['ndarray_masked_mask']
    else:
        d = h_group.create_dataset('data_%i' % call_id, data=py_obj, **kwargs)
        d.attrs["type"] = ['ndarray']


def create_sparse_dataset(py_obj, h_group, call_id=0, **kwargs):
    """ dumps an sparse array to h5py file

    Args:
        py_obj: python object to dump; should be a numpy array or np.ma.array (masked)
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the iterable.
    """
    h_sparsegroup = h_group.create_group('data_%i' % call_id)
    data    = h_sparsegroup.create_dataset('data',    data=py_obj.data, **kwargs)
    indices = h_sparsegroup.create_dataset('indices', data=py_obj.indices, **kwargs)
    indptr  = h_sparsegroup.create_dataset('indptr',  data=py_obj.indptr, **kwargs)
    shape   = h_sparsegroup.create_dataset('shape',   data=py_obj.shape, **kwargs)

    if isinstance(py_obj, type(sparse.csr_matrix([0]))):
        type_str = 'csr'
    elif isinstance(py_obj, type(sparse.csc_matrix([0]))):
        type_str = 'csc'
    elif isinstance(py_obj, type(sparse.bsr_matrix([0]))):
        type_str = 'bsr'

    h_sparsegroup.attrs["type"] = ['%s_matrix' % type_str]
    data.attrs["type"] = ["%s_matrix_data" % type_str]
    indices.attrs["type"] = ["%s_matrix_indices" % type_str]
    indptr.attrs["type"] = ["%s_matrix_indptr" % type_str]
    shape.attrs["type"] = ["%s_matrix_shape" % type_str]