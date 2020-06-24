# encoding: utf-8
"""
# load_numpy.py

Utilities and dump / load handlers for handling numpy and scipy arrays

"""

# %% IMPORTS
# Package imports
import numpy as np
import dill as pickle

# hickle imports
from hickle.helpers import get_type_and_data
from hickle.hickle import _dump


# %% FUNCTION DEFINITIONS
def check_is_numpy_array(py_obj):
    """ Check if a python object is a numpy array (masked or regular)

    Args:
        py_obj: python object to check whether it is a numpy array

    Returns
        is_numpy (bool): Returns True if it is a numpy array, else False if it
            isn't
    """

    return(isinstance(py_obj, np.ndarray))


def create_np_scalar_dataset(py_obj, h_group, name, **kwargs):
    """ dumps an np dtype object to h5py file

    Args:
        py_obj: python object to dump; should be a numpy scalar, e.g.
            np.float16(1)
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the
            iterable.
    """

    d = h_group.create_dataset(name, data=py_obj, **kwargs)

    d.attrs["np_dtype"] = bytes(str(d.dtype), 'ascii')
    return(d)


def create_np_dtype(py_obj, h_group, name, **kwargs):
    """ dumps an np dtype object to h5py file

    Args:
        py_obj: python object to dump; should be a numpy dtype, e.g.
            np.float16
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the
            iterable.
    """
    d = h_group.create_dataset(name, data=str(py_obj), **kwargs)
    return(d)


def create_np_array_dataset(py_obj, h_group, name, **kwargs):
    """ dumps an ndarray object to h5py file

    Args:
        py_obj: python object to dump; should be a numpy array or np.ma.array
            (masked)
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the
            iterable.
    """

    # Obtain dtype of py_obj
    dtype = str(py_obj.dtype)

    # Check if py_obj contains strings
    if '<U' in dtype:
        # If so, convert the array to one with bytes
        py_obj = np.array(py_obj, dtype=dtype.replace('<U', '|S'))

    if isinstance(py_obj, np.ma.core.MaskedArray):
        d = h_group.create_dataset(name, data=py_obj, **kwargs)
        m = h_group.create_dataset('%s_mask' % name, data=py_obj.mask,
                                   **kwargs)
        m.attrs['type'] = np.array(pickle.dumps(py_obj.mask.__class__))
        m.attrs['base_type'] = b'ndarray_masked_mask'
    # Check if py_obj contains an object not understood by NumPy
    elif 'object' in dtype:
        # If so, convert py_obj to list
        py_obj = py_obj.tolist()

        # Check if py_obj is a list
        if isinstance(py_obj, list):
            # If so, dump py_obj into the current group
            _dump(py_obj, h_group, name, **kwargs)
            d = h_group[name]
            d.attrs['type'] = np.array(pickle.dumps(np.array))
        else:
            # If not, create a new group and dump py_obj into that
            d = h_group.create_group(name)
            _dump(py_obj, d, **kwargs)
            d.attrs['type'] = np.array(pickle.dumps(lambda x: np.array(x[0])))
    else:
        d = h_group.create_dataset(name, data=py_obj, **kwargs)
    d.attrs['np_dtype'] = bytes(dtype, 'ascii')
    return(d)


#####################
# Lookup dictionary #
#####################

def load_np_dtype_dataset(h_node):
    _, _, data = get_type_and_data(h_node)
    data = np.dtype(data)
    return data


def load_np_scalar_dataset(h_node):
    _, _, data = get_type_and_data(h_node)
    return data


def load_ndarray_dataset(h_node):
    _, _, data = get_type_and_data(h_node)
    dtype = h_node.attrs['np_dtype']
    return np.array(data, copy=False, dtype=dtype)


def load_ndarray_masked_dataset(h_node):
    _, _, data = get_type_and_data(h_node)
    dtype = h_node.attrs['np_dtype']
    try:
        mask_path = h_node.name + "_mask"
        h_root = h_node.parent
        mask = h_root.get(mask_path)[:]
    except (ValueError, IndexError):
        mask = h_root.get(mask_path)
    data = np.ma.array(data, mask=mask, dtype=dtype)
    return data


# %% REGISTERS
class_register = [
    [np.dtype, b"np_dtype", create_np_dtype, load_np_dtype_dataset],
    [np.ndarray, b"ndarray", create_np_array_dataset, load_ndarray_dataset,
     check_is_numpy_array],
    [np.ma.core.MaskedArray, b"ndarray_masked_data", create_np_array_dataset,
     load_ndarray_masked_dataset, check_is_numpy_array],
    [np.number, b"np_scalar", create_np_scalar_dataset,
     load_np_scalar_dataset]]

exclude_register = [b"ndarray_masked_mask"]
