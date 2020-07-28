# encoding: utf-8
"""
# hickle.py

Created by Danny Price 2016-02-03.

Hickle is a HDF5 based clone of Pickle. Instead of serializing to a pickle
file, Hickle dumps to a HDF5 file. It is designed to be as similar to pickle in
usage as possible, providing a load() and dump() function.

## Notes

Hickle has two main advantages over Pickle:
1) LARGE PICKLE HANDLING. Unpickling a large pickle is slow, as the Unpickler
reads the entire pickle thing and loads it into memory. In comparison, HDF5
files are designed for large datasets. Things are only loaded when accessed.

2) CROSS PLATFORM SUPPORT. Attempting to unpickle a pickle pickled on Windows
on Linux and vice versa is likely to fail with errors like "Insecure string
pickle". HDF5 files will load fine, as long as both machines have
h5py installed.

"""

from __future__ import absolute_import, division, print_function
import sys
import os
from pkg_resources import get_distribution, DistributionNotFound
from ast import literal_eval

import numpy as np
import h5py as h5


from .__version__ import __version__
from .helpers import get_type, sort_keys, check_is_iterable, check_iterable_item_type
from .lookup import (types_dict, hkl_types_dict, types_not_to_sort,
    container_types_dict, container_key_types_dict, check_is_ndarray_like)

try:
    from exceptions import Exception
    from types import NoneType
except ImportError:
    pass        # above imports will fail in python3

from six import PY2, PY3, string_types, integer_types
import io

# Make several aliases for Python2/Python3 compatibility
if PY3:
    file = io.TextIOWrapper

# Import dill as pickle
import dill as pickle

try:
    from pathlib import Path
    string_like_types = string_types + (Path,)
except ImportError:
    # Python 2 does not have pathlib
    string_like_types = string_types

import warnings

##################
# Error handling #
##################

class FileError(Exception):
    """ An exception raised if the file is fishy """
    def __init__(self):
        return

    def __str__(self):
        return ("Cannot open file. Please pass either a filename "
                "string, a file object, or a h5py.File")


class ClosedFileError(Exception):
    """ An exception raised if the file is fishy """
    def __init__(self):
        return

    def __str__(self):
        return ("HDF5 file has been closed. Please pass either "
                "a filename string, a file object, or an open h5py.File")


class NoMatchError(Exception):
    """ An exception raised if the object type is not understood (or
    supported)"""
    def __init__(self):
        return

    def __str__(self):
        return ("Error: this type of python object cannot be converted into a "
                "hickle.")


class ToDoError(Exception):
    """ An exception raised for non-implemented functionality"""
    def __init__(self):
        return

    def __str__(self):
        return "Error: this functionality hasn't been implemented yet."


class SerializedWarning(UserWarning):
    """ An object type was not understood

    The data will be serialized using pickle.
    """
    pass


######################
# H5PY file wrappers #
######################

class H5GroupWrapper(h5.Group):
    """ Group wrapper that provides a track_times kwarg.

    track_times is a boolean flag that can be set to False, so that two
    files created at different times will have identical MD5 hashes.
    """
    def create_dataset(self, *args, **kwargs):
        kwargs['track_times'] = getattr(self, 'track_times', True)
        return super(H5GroupWrapper, self).create_dataset(*args, **kwargs)

    def create_group(self, *args, **kwargs):
        group = super(H5GroupWrapper, self).create_group(*args, **kwargs)
        group.__class__ = H5GroupWrapper
        group.track_times = getattr(self, 'track_times', True)
        return group


class H5FileWrapper(h5.File):
    """ Wrapper for h5py File that provides a track_times kwarg.

    track_times is a boolean flag that can be set to False, so that two
    files created at different times will have identical MD5 hashes.
    """
    def create_dataset(self, *args, **kwargs):
        kwargs['track_times'] = getattr(self, 'track_times', True)
        return super(H5FileWrapper, self).create_dataset(*args, **kwargs)

    def create_group(self, *args, **kwargs):
        group = super(H5FileWrapper, self).create_group(*args, **kwargs)
        group.__class__ = H5GroupWrapper
        group.track_times = getattr(self, 'track_times', True)
        return group


def file_opener(f, mode='r', track_times=True):
    """ A file opener helper function with some error handling.  This can open
    files through a file object, a h5py file, or just the filename.

    Args:
        f (file, h5py.File, or string): File-identifier, e.g. filename or file object.
        mode (str): File open mode. Only required if opening by filename string.
        track_times (bool): Track time in HDF5; turn off if you want hickling at
                 different times to produce identical files (e.g. for MD5 hash check).

    """

    # Assume that we will have to close the file after dump or load
    close_flag = True

    # Were we handed a file object or just a file name string?
    if isinstance(f, (file, io.TextIOWrapper, io.BufferedWriter)):
        filename, mode = f.name, f.mode
        f.close()
        mode = mode.replace('b', '')
        h5f = h5.File(filename, mode)
    elif isinstance(f, string_like_types):
        filename = f
        h5f = h5.File(filename, mode)
    elif isinstance(f, (H5FileWrapper, h5._hl.files.File)):
        try:
            filename = f.filename
        except ValueError:
            raise ClosedFileError
        h5f = f
        # Since this file was already open, do not close the file afterward
        close_flag = False
    else:
        print(f.__class__)
        raise FileError

    h5f.__class__ = H5FileWrapper
    h5f.track_times = track_times
    return(h5f, close_flag)


###########
# DUMPERS #
###########


def _dump(py_obj, h_group, call_id=0, **kwargs):
    """ Dump a python object to a group within a HDF5 file.

    This function is called recursively by the main dump() function.

    Args:
        py_obj: python object to dump.
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the iterable.
    """

    # Get list of dumpable dtypes
    dumpable_dtypes = []
    for lst in [[bool, complex, bytes, float], string_types, integer_types]:
        dumpable_dtypes.extend(lst)

    # Firstly, check if item is a numpy array. If so, just dump it.
    if check_is_ndarray_like(py_obj):
        create_hkl_dataset(py_obj, h_group, call_id, **kwargs)

    # Next, check if item is a dict
    elif isinstance(py_obj, dict):
        create_hkl_dataset(py_obj, h_group, call_id, **kwargs)

    # If not, check if item is iterable
    elif check_is_iterable(py_obj):
        item_type = check_iterable_item_type(py_obj)

        # item_type == False implies multiple types. Create a dataset
        if item_type is False:
            h_subgroup = create_hkl_group(py_obj, h_group, call_id)
            for ii, py_subobj in enumerate(py_obj):
                _dump(py_subobj, h_subgroup, call_id=ii, **kwargs)

        # otherwise, subitems have same type. Check if subtype is an iterable
        # (e.g. list of lists), or not (e.g. list of ints, which should be treated
        # as a single dataset).
        else:
            if item_type in dumpable_dtypes:
                create_hkl_dataset(py_obj, h_group, call_id, **kwargs)
            else:
                h_subgroup = create_hkl_group(py_obj, h_group, call_id)
                for ii, py_subobj in enumerate(py_obj):
                    _dump(py_subobj, h_subgroup, call_id=ii, **kwargs)

    # item is not iterable, so create a dataset for it
    else:
        create_hkl_dataset(py_obj, h_group, call_id, **kwargs)


def dump(py_obj, file_obj, mode='w', track_times=True, path='/', **kwargs):
    """ Write a pickled representation of obj to the open file object file.

    Args:
    obj (object): python object o store in a Hickle
    file: file object, filename string, or h5py.File object
            file in which to store the object. A h5py.File or a filename is also
            acceptable.
    mode (str): optional argument, 'r' (read only), 'w' (write) or 'a' (append).
            Ignored if file is a file object.
    compression (str): optional argument. Applies compression to dataset. Options: None, gzip,
            lzf (+ szip, if installed)
    track_times (bool): optional argument. If set to False, repeated hickling will produce
            identical files.
    path (str): path within hdf5 file to save data to. Defaults to root /
    """

    # Make sure that file is not closed unless modified
    # This is to avoid trying to close a file that was never opened
    close_flag = False

    try:
        # Open the file
        h5f, close_flag = file_opener(file_obj, mode, track_times)
        h5f.attrs["CLASS"] = b'hickle'
        h5f.attrs["VERSION"] = __version__
        h5f.attrs["type"] = [b'hickle']
        # Log which version of python was used to generate the hickle file
        pv = sys.version_info
        py_ver = "%i.%i.%i" % (pv[0], pv[1], pv[2])
        h5f.attrs["PYTHON_VERSION"] = py_ver

        h_root_group = h5f.get(path)

        if h_root_group is None:
            h_root_group = h5f.create_group(path)
            h_root_group.attrs["type"] = [b'hickle']

        _dump(py_obj, h_root_group, **kwargs)
    except NoMatchError:
        fname = h5f.filename
        h5f.close()
        try:
            os.remove(fname)
        except OSError:
            warnings.warn("Dump failed. Could not remove %s" % fname)
        finally:
            raise NoMatchError
    finally:
        # Close the file if requested.
        # Closing a file twice will not cause any problems
        if close_flag:
            h5f.close()


def create_dataset_lookup(py_obj):
    """ What type of object are we trying to pickle?  This is a python
    dictionary based equivalent of a case statement.  It returns the correct
    helper function for a given data type.

    Args:
        py_obj: python object to look-up what function to use to dump to disk

    Returns:
        match: function that should be used to dump data to a new dataset
    """
    t = type(py_obj)
    types_lookup = {dict: create_dict_dataset}
    types_lookup.update(types_dict)

    match = types_lookup.get(t, no_match)

    return match



def create_hkl_dataset(py_obj, h_group, call_id=0, **kwargs):
    """ Create a dataset within the hickle HDF5 file

    Args:
        py_obj: python object to dump.
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the iterable.

    """
    #lookup dataset creator type based on python object type
    create_dataset = create_dataset_lookup(py_obj)

    # do the creation
    create_dataset(py_obj, h_group, call_id, **kwargs)


def create_hkl_group(py_obj, h_group, call_id=0):
    """ Create a new group within the hickle file

    Args:
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the iterable.

    """
    h_subgroup = h_group.create_group('data_%i' % call_id)
    h_subgroup.attrs['type'] = [str(type(py_obj)).encode('ascii', 'ignore')]
    return h_subgroup


def create_dict_dataset(py_obj, h_group, call_id=0, **kwargs):
    """ Creates a data group for each key in dictionary

    Notes:
        This is a very important function which uses the recursive _dump
        method to build up hierarchical data models stored in the HDF5 file.
        As this is critical to functioning, it is kept in the main hickle.py
        file instead of in the loaders/ directory.

    Args:
        py_obj: python object to dump; should be dictionary
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the iterable.
    """
    h_dictgroup = h_group.create_group('data_%i' % call_id)
    h_dictgroup.attrs['type'] = [str(type(py_obj)).encode('ascii', 'ignore')]

    for key, py_subobj in py_obj.items():
        if isinstance(key, string_types):
            h_subgroup = h_dictgroup.create_group("%r" % (key))
        else:
            h_subgroup = h_dictgroup.create_group(str(key))
        h_subgroup.attrs["type"] = [b'dict_item']

        h_subgroup.attrs["key_type"] = [str(type(key)).encode('ascii', 'ignore')]

        _dump(py_subobj, h_subgroup, call_id=0, **kwargs)


def no_match(py_obj, h_group, call_id=0, **kwargs):
    """ If no match is made, raise an exception

    Args:
        py_obj: python object to dump; default if item is not matched.
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the iterable.
    """
    pickled_obj = pickle.dumps(py_obj)
    d = h_group.create_dataset('data_%i' % call_id, data=[pickled_obj])
    d.attrs["type"] = [b'pickle']

    warnings.warn("%s type not understood, data have been serialized" % type(py_obj),
                  SerializedWarning)



#############
## LOADERS ##
#############

class PyContainer(list):
    """ A group-like object into which to load datasets.

    In order to build up a tree-like structure, we need to be able
    to load datasets into a container with an append() method.
    Python tuples and sets do not allow this. This class provides
    a list-like object that be converted into a list, tuple, set or dict.
    """
    def __init__(self):
        super(PyContainer, self).__init__()
        self.container_type = None
        self.name = None
        self.key_type = None

    def convert(self):
        """ Convert from PyContainer to python core data type.

        Returns: self, either as a list, tuple, set or dict
                 (or other type specified in lookup.py)
        """

        if self.container_type in container_types_dict.keys():
            convert_fn = container_types_dict[self.container_type]
            return convert_fn(self)
        if self.container_type == str(dict).encode('ascii', 'ignore'):
            keys = []
            for item in self:
                key = item.name.split('/')[-1]
                key_type = item.key_type[0]
                if key_type in container_key_types_dict.keys():
                    to_type_fn = container_key_types_dict[key_type]
                    key = to_type_fn(key)
                keys.append(key)

            items = [item[0] for item in self]
            return dict(zip(keys, items))
        else:
            return self

def no_match_load(key):
    """ If no match is made when loading, need to raise an exception
    """
    raise RuntimeError("Cannot load %s data type" % key)
    #pass

def load_dataset_lookup(key):
    """ What type of object are we trying to unpickle?  This is a python
    dictionary based equivalent of a case statement.  It returns the type
    a given 'type' keyword in the hickle file.

    Args:
        py_obj: python object to look-up what function to use to dump to disk

    Returns:
        match: function that should be used to dump data to a new dataset
    """

    match = hkl_types_dict.get(key, no_match_load)

    return match

def load(fileobj, path='/', safe=True):
    """ Load a hickle file and reconstruct a python object

    Args:
        fileobj: file object, h5py.File, or filename string
            safe (bool): Disable automatic depickling of arbitrary python objects.
            DO NOT set this to False unless the file is from a trusted source.
            (see http://www.cs.jhu.edu/~s/musings/pickle.html for an explanation)

        path (str): path within hdf5 file to save data to. Defaults to root /
    """

    # Make sure that the file is not closed unless modified
    # This is to avoid trying to close a file that was never opened
    close_flag = False

    try:
        h5f, close_flag = file_opener(fileobj)
        h_root_group = h5f.get(path)
        try:
            assert 'CLASS' in h5f.attrs.keys()
            assert 'VERSION' in h5f.attrs.keys()
            VER = h5f.attrs['VERSION']
            try:
                VER_MAJOR = int(VER)
            except ValueError:
                VER_MAJOR = int(VER[0])
            if VER_MAJOR == 1:
                if PY2:
                    warnings.warn("Hickle file versioned as V1, attempting legacy loading...")
                    from . import hickle_legacy
                    return hickle_legacy.load(fileobj, safe)
                else:
                    raise RuntimeError("Cannot open file. This file was likely"
                                       " created with Python 2 and an old hickle version.")
            elif VER_MAJOR == 2:
                if PY2:
                    warnings.warn("Hickle file appears to be old version (v2), attempting "
                                  "legacy loading...")
                    from . import hickle_legacy2
                    return hickle_legacy2.load(fileobj, path=path, safe=safe)
                else:
                    raise RuntimeError("Cannot open file. This file was likely"
                                       " created with Python 2 and an old hickle version.")
            # There is an unfortunate period of time where hickle 2.1.0 claims VERSION = int(3)
            # For backward compatibility we really need to catch this.
            # Actual hickle v3 files are versioned as A.B.C (e.g. 3.1.0)
            elif VER_MAJOR == 3 and VER == VER_MAJOR:
                if PY2:
                    warnings.warn("Hickle file appears to be old version (v2.1.0), attempting "
                                  "legacy loading...")
                    from . import hickle_legacy2
                    return hickle_legacy2.load(fileobj, path=path, safe=safe)
                else:
                    raise RuntimeError("Cannot open file. This file was likely"
                                       " created with Python 2 and an old hickle version.")
            elif VER_MAJOR >= 3:
                py_container = PyContainer()
                py_container.container_type = 'hickle'
                py_container = _load(py_container, h_root_group)
                return py_container[0][0]

        except AssertionError:
            if PY2:
                warnings.warn("Hickle file is not versioned, attempting legacy loading...")
                from . import hickle_legacy
                return hickle_legacy.load(fileobj, safe)
            else:
                raise RuntimeError("Cannot open file. This file was likely"
                                   " created with Python 2 and an old hickle version.")
    finally:
        # Close the file if requested.
        # Closing a file twice will not cause any problems
        if close_flag:
            h5f.close()

def load_dataset(h_node):
    """ Load a dataset, converting into its correct python type

    Args:
        h_node (h5py dataset): h5py dataset object to read

    Returns:
        data: reconstructed python object from loaded data
    """
    py_type = get_type(h_node)

    try:
        load_fn = load_dataset_lookup(py_type)
        return load_fn(h_node)
    except:
        raise
        #raise RuntimeError("Hickle type %s not understood." % py_type)

def _load(py_container, h_group):
    """ Load a hickle file

    Recursive funnction to load hdf5 data into a PyContainer()

    Args:
        py_container (PyContainer): Python container to load data into
        h_group (h5 group or dataset): h5py object, group or dataset, to spider
                and load all datasets.
    """

    group_dtype   = h5._hl.group.Group
    dataset_dtype = h5._hl.dataset.Dataset

    #either a file, group, or dataset
    if isinstance(h_group, (H5FileWrapper, group_dtype)):

        py_subcontainer = PyContainer()
        try:
            py_subcontainer.container_type = bytes(h_group.attrs['type'][0])
        except KeyError:
            raise
            #py_subcontainer.container_type = ''
        py_subcontainer.name = h_group.name

        if py_subcontainer.container_type == b'dict_item':
            py_subcontainer.key_type = h_group.attrs['key_type']

        if py_subcontainer.container_type not in types_not_to_sort:
            h_keys = sort_keys(h_group.keys())
        else:
            h_keys = h_group.keys()

        for h_name in h_keys:
            h_node = h_group[h_name]
            py_subcontainer = _load(py_subcontainer, h_node)

        sub_data = py_subcontainer.convert()
        py_container.append(sub_data)

    else:
        # must be a dataset
        subdata = load_dataset(h_group)
        py_container.append(subdata)

    return py_container
