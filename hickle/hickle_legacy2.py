# encoding: utf-8
"""
# hickle_legacy2.py

Created by Danny Price 2016-02-03.

This is a legacy handler, for hickle v2 files.
If V3 reading fails, this will be called as a fail-over.

"""

import os
import numpy as np
import h5py as h5
import re

try:
    from exceptions import Exception
    from types import NoneType
except ImportError:
    pass        # above imports will fail in python3

import warnings
__version__ = "2.0.4"
__author__ = "Danny Price"


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
    # Were we handed a file object or just a file name string?
    if isinstance(f, file):
        filename, mode = f.name, f.mode
        f.close()
        h5f = h5.File(filename, mode)
    elif isinstance(f, str) or isinstance(f, unicode):
        filename = f
        h5f = h5.File(filename, mode)
    elif isinstance(f, H5FileWrapper) or isinstance(f, h5._hl.files.File):
        try:
            filename = f.filename
        except ValueError:
            raise ClosedFileError()
        h5f = f
    else:
        print(type(f))
        raise FileError

    h5f.__class__ = H5FileWrapper
    h5f.track_times = track_times
    return h5f


###########
# DUMPERS #
###########

def check_is_iterable(py_obj):
    """ Check whether a python object is iterable.

    Note: this treats unicode and string as NON ITERABLE

    Args:
        py_obj: python object to test

    Returns:
        iter_ok (bool): True if item is iterable, False is item is not
    """
    if type(py_obj) in (str, unicode):
        return False
    try:
        iter(py_obj)
        return True
    except TypeError:
        return False


def check_iterable_item_type(iter_obj):
    """ Check if all items within an iterable are the same type.

    Args:
        iter_obj: iterable object

    Returns:
        iter_type: type of item contained within the iterable. If
                   the iterable has many types, a boolean False is returned instead.

    References:
    http://stackoverflow.com/questions/13252333/python-check-if-all-elements-of-a-list-are-the-same-type
    """
    iseq = iter(iter_obj)
    first_type = type(next(iseq))
    return first_type if all((type(x) is first_type) for x in iseq) else False


def check_is_numpy_array(py_obj):
    """ Check if a python object is a numpy array (masked or regular)

    Args:
        py_obj: python object to check whether it is a numpy array

    Returns
        is_numpy (bool): Returns True if it is a numpy array, else False if it isn't
    """

    is_numpy = type(py_obj) in (type(np.array([1])), type(np.ma.array([1])))

    return is_numpy


def _dump(py_obj, h_group, call_id=0, **kwargs):
    """ Dump a python object to a group within a HDF5 file.

    This function is called recursively by the main dump() function.

    Args:
        py_obj: python object to dump.
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the iterable.
    """

    dumpable_dtypes = set([bool, int, float, long, complex, str, unicode])

    # Firstly, check if item is a numpy array. If so, just dump it.
    if check_is_numpy_array(py_obj):
        create_hkl_dataset(py_obj, h_group, call_id, **kwargs)

    # next, check if item is iterable
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
                    #print py_subobj, h_subgroup, ii
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

    try:
        # Open the file
        h5f = file_opener(file_obj, mode, track_times)
        h5f.attrs["CLASS"] = 'hickle'
        h5f.attrs["VERSION"] = 2
        h5f.attrs["type"] = ['hickle']

        h_root_group = h5f.get(path)

        if h_root_group is None:
            h_root_group = h5f.create_group(path)
            h_root_group.attrs["type"] = ['hickle']

        _dump(py_obj, h_root_group, **kwargs)
        h5f.close()
    except NoMatchError:
        fname = h5f.filename
        h5f.close()
        try:
            os.remove(fname)
        except OSError:
            warnings.warn("Dump failed. Could not remove %s" % fname)
        finally:
            raise NoMatchError


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

    types = {
        dict: create_dict_dataset,
        list: create_listlike_dataset,
        tuple: create_listlike_dataset,
        set: create_listlike_dataset,
        str: create_stringlike_dataset,
        unicode: create_stringlike_dataset,
        int: create_python_dtype_dataset,
        float: create_python_dtype_dataset,
        long: create_python_dtype_dataset,
        bool: create_python_dtype_dataset,
        complex: create_python_dtype_dataset,
        NoneType: create_none_dataset,
        np.ndarray: create_np_array_dataset,
        np.ma.core.MaskedArray: create_np_array_dataset,
        np.float16: create_np_dtype_dataset,
        np.float32: create_np_dtype_dataset,
        np.float64: create_np_dtype_dataset,
        np.int8: create_np_dtype_dataset,
        np.int16: create_np_dtype_dataset,
        np.int32: create_np_dtype_dataset,
        np.int64: create_np_dtype_dataset,
        np.uint8: create_np_dtype_dataset,
        np.uint16: create_np_dtype_dataset,
        np.uint32: create_np_dtype_dataset,
        np.uint64: create_np_dtype_dataset,
        np.complex64: create_np_dtype_dataset,
        np.complex128: create_np_dtype_dataset
    }

    match = types.get(t, no_match)
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
    h_subgroup.attrs["type"] = [str(type(py_obj))]
    return h_subgroup


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


def create_np_dtype_dataset(py_obj, h_group, call_id=0, **kwargs):
    """ dumps an np dtype object to h5py file

    Args:
        py_obj: python object to dump; should be a numpy scalar, e.g. np.float16(1)
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the iterable.
    """
    d = h_group.create_dataset('data_%i' % call_id, data=py_obj, **kwargs)
    d.attrs["type"] = ['np_dtype']
    d.attrs["np_dtype"] = str(d.dtype)


def create_python_dtype_dataset(py_obj, h_group, call_id=0, **kwargs):
    """ dumps a python dtype object to h5py file

    Args:
        py_obj: python object to dump; should be a python type (int, float, bool etc)
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the iterable.
    """
    d = h_group.create_dataset('data_%i' % call_id, data=py_obj,
                               dtype=type(py_obj), **kwargs)
    d.attrs["type"] = ['python_dtype']
    d.attrs['python_subdtype'] = str(type(py_obj))


def create_dict_dataset(py_obj, h_group, call_id=0, **kwargs):
    """ Creates a data group for each key in dictionary

    Args:
        py_obj: python object to dump; should be dictionary
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the iterable.
    """
    h_dictgroup = h_group.create_group('data_%i' % call_id)
    h_dictgroup.attrs["type"] = ['dict']
    for key, py_subobj in py_obj.items():
        h_subgroup = h_dictgroup.create_group(key)
        h_subgroup.attrs["type"] = ['dict_item']
        _dump(py_subobj, h_subgroup, call_id=0, **kwargs)


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


def no_match(py_obj, h_group, call_id=0, **kwargs):
    """ If no match is made, raise an exception

    Args:
        py_obj: python object to dump; default if item is not matched.
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the iterable.
    """
    try:
        import dill as cPickle
    except ImportError:
        import cPickle

    pickled_obj = cPickle.dumps(py_obj)
    d = h_group.create_dataset('data_%i' % call_id, data=[pickled_obj])
    d.attrs["type"] = ['pickle']

    warnings.warn("%s type not understood, data have been "
                  "serialized" % type(py_obj))


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

    def convert(self):
        """ Convert from PyContainer to python core data type.

        Returns: self, either as a list, tuple, set or dict
        """
        if self.container_type == "<type 'list'>":
            return list(self)
        if self.container_type == "<type 'tuple'>":
            return tuple(self)
        if self.container_type == "<type 'set'>":
            return set(self)
        if self.container_type == "dict":
            keys = [str(item.name.split('/')[-1]) for item in self]
            items = [item[0] for item in self]
            return dict(zip(keys, items))
        else:
            return self


def load(fileobj, path='/', safe=True):
    """ Load a hickle file and reconstruct a python object

    Args:
        fileobj: file object, h5py.File, or filename string
            safe (bool): Disable automatic depickling of arbitrary python objects.
            DO NOT set this to False unless the file is from a trusted source.
            (see http://www.cs.jhu.edu/~s/musings/pickle.html for an explanation)

        path (str): path within hdf5 file to save data to. Defaults to root /
    """

    try:
        h5f = file_opener(fileobj)
        h_root_group = h5f.get(path)

        try:
            assert 'CLASS' in h5f.attrs.keys()
            assert 'VERSION' in h5f.attrs.keys()
            py_container = PyContainer()
            py_container.container_type = 'hickle'
            py_container = _load(py_container, h_root_group)
            return py_container[0][0]
        except AssertionError:
            import hickle_legacy
            return hickle_legacy.load(fileobj, safe)
    finally:
        if 'h5f' in locals():
            h5f.close()


def load_dataset(h_node):
    """ Load a dataset, converting into its correct python type

    Args:
        h_node (h5py dataset): h5py dataset object to read

    Returns:
        data: reconstructed python object from loaded data
    """
    py_type = h_node.attrs["type"][0]

    if h_node.shape == ():
        data = h_node.value
    else:
        data  = h_node[:]

    if py_type == "<type 'list'>":
        #print self.name
        return list(data)
    elif py_type == "<type 'tuple'>":
        return tuple(data)
    elif py_type == "<type 'set'>":
        return set(data)
    elif py_type == "np_dtype":
        subtype = h_node.attrs["np_dtype"]
        data = np.array(data, dtype=subtype)
        return data
    elif py_type == 'ndarray':
        return np.array(data)
    elif py_type == 'ndarray_masked_data':
        try:
            mask_path = h_node.name + "_mask"
            h_root = h_node.parent
            mask = h_root.get(mask_path)[:]
        except IndexError:
            mask = h_root.get(mask_path)
        except ValueError:
            mask = h_root.get(mask_path)
        data = np.ma.array(data, mask=mask)
        return data
    elif py_type == 'python_dtype':
        subtype = h_node.attrs["python_subdtype"]
        type_dict = {
            "<type 'int'>": int,
            "<type 'float'>": float,
            "<type 'long'>": long,
            "<type 'bool'>": bool,
            "<type 'complex'>": complex
        }
        tcast = type_dict.get(subtype)
        return tcast(data)
    elif py_type == 'string':
        return str(data[0])
    elif py_type == 'unicode':
        return unicode(data[0])
    elif py_type == 'none':
        return None
    else:
        print(h_node.name, py_type, h_node.attrs.keys())
        return data


def sort_keys(key_list):
    """ Take a list of strings and sort it by integer value within string

    Args:
        key_list (list): List of keys

    Returns:
        key_list_sorted (list): List of keys, sorted by integer
    """
    to_int = lambda x: int(re.search('\d+', x).group(0))
    keys_by_int = sorted([(to_int(key), key) for key in key_list])
    return [ii[1] for ii in keys_by_int]


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
    if isinstance(h_group, H5FileWrapper) or isinstance(h_group, group_dtype):
        py_subcontainer = PyContainer()
        py_subcontainer.container_type = h_group.attrs['type'][0]
        py_subcontainer.name = h_group.name

        if py_subcontainer.container_type != 'dict':
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

    #print h_group.name, py_container
    return py_container
