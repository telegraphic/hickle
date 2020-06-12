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


# %% IMPORTS
# Built-in imports
import io
from pathlib import Path
import sys
import warnings

# Package imports
import dill as pickle
import h5py as h5
import numpy as np

# hickle imports
from hickle import __version__
from hickle.helpers import (
    get_type, sort_keys, check_is_iterable, check_iterable_item_type)
from hickle.lookup import (
    types_dict, hkl_types_dict, types_not_to_sort, dict_key_types_dict,
    check_is_ndarray_like, load_loader)

# All declaration
__all__ = ['dump', 'load']


# %% CLASS DEFINITIONS
##################
# Error handling #
##################

class FileError(Exception):
    """ An exception raised if the file is fishy """
    pass


class ClosedFileError(Exception):
    """ An exception raised if the file is fishy """
    pass


class ToDoError(Exception):     # pragma: no cover
    """ An exception raised for non-implemented functionality"""
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

    def create_group(self, *args, **kwargs):
        group = super(H5FileWrapper, self).create_group(*args, **kwargs)
        group.__class__ = H5GroupWrapper
        group.track_times = getattr(self, 'track_times', True)
        return group


# %% FUNCTION DEFINITIONS
def file_opener(f, path, mode='r', track_times=True):
    """
    A file opener helper function with some error handling.
    This can open files through a file object, an h5py file, or just the
    filename.

    Parameters
    ----------
    f : file object, str or :obj:`~h5py.Group` object
        File to open for dumping or loading purposes.
        If str, `file_obj` provides the path of the HDF5-file that must be
        used.
        If :obj:`~h5py._hl.group.Group`, the group (or file) in an open
        HDF5-file that must be used.
    path : str
        Path within HDF5-file or group to dump to/load from.
    mode : str, optional
        Accepted values are 'r' (read only), 'w' (write; default) or 'a'
        (append).
        Ignored if file is a file object.
    track_times : bool, optional
        If set to *True* (default), repeated hickling will produce different
        files.

    """

    # Assume that we will have to close the file after dump or load
    close_flag = True

    # Make sure that the given path always starts with '/'
    if not path.startswith('/'):
        path = '/%s' % (path)

    # Were we handed a file object or just a file name string?
    if isinstance(f, (io.TextIOWrapper, io.BufferedWriter)):
        filename, mode = f.name, f.mode
        f.close()
        mode = mode.replace('b', '')
        h5f = h5.File(filename, mode)
    elif isinstance(f, (str, Path)):
        filename = f
        h5f = h5.File(filename, mode)
    elif isinstance(f, h5._hl.group.Group):
        try:
            filename = f.file.filename
        except ValueError:
            raise ClosedFileError("HDF5 file has been closed. Please pass "
                                  "either a filename string, a file object, or"
                                  "an open HDF5-file")
        path = ''.join([f.name, path])
        h5f = f

        if path.endswith('/'):
            path = path[:-1]

        # Since this file was already open, do not close the file afterward
        close_flag = False

    else:
        print(f.__class__)
        raise FileError("Cannot open file. Please pass either a filename "
                        "string, a file object, or a h5py.File")

    if isinstance(h5f, h5._hl.files.File):
        h5f.__class__ = H5FileWrapper
    else:
        h5f.__class__ = H5GroupWrapper
    h5f.track_times = track_times
    return(h5f, path, close_flag)


###########
# DUMPERS #
###########

# Get list of dumpable dtypes
dumpable_dtypes = [bool, complex, bytes, float, int, str]


def _dump(py_obj, h_group, call_id=None, **kwargs):
    """ Dump a python object to a group within a HDF5 file.

    This function is called recursively by the main dump() function.

    Args:
        py_obj: python object to dump.
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the
            iterable.
    """

    # Check if we have a unloaded loader for the provided py_obj
    load_loader(py_obj)

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
                if len(py_obj) == 1:
                    ii = None
                _dump(py_subobj, h_subgroup, call_id=ii, **kwargs)

        # otherwise, subitems have same type. Check if subtype is an iterable
        # (e.g. list of lists), or not (e.g. list of ints, which should be
        # treated as a single dataset).
        else:
            if item_type in dumpable_dtypes:
                create_hkl_dataset(py_obj, h_group, call_id, **kwargs)
            else:
                h_subgroup = create_hkl_group(py_obj, h_group, call_id)
                for ii, py_subobj in enumerate(py_obj):
                    if len(py_obj) == 1:
                        ii = None
                    _dump(py_subobj, h_subgroup, call_id=ii, **kwargs)

    # item is not iterable, so create a dataset for it
    else:
        create_hkl_dataset(py_obj, h_group, call_id, **kwargs)


def dump(py_obj, file_obj, mode='w', path='/', track_times=True, **kwargs):
    """
    Write a hickled representation of `py_obj` to the provided `file_obj`.

    Parameters
    ----------
    py_obj : object
        Python object to hickle to HDF5.
    file_obj : file object, str or :obj:`~h5py.Group` object
        File in which to store the object.
        If str, `file_obj` provides the path of the HDF5-file that must be
        used.
        If :obj:`~h5py._hl.group.Group`, the group (or file) in an open
        HDF5-file that must be used.
    mode : str, optional
        Accepted values are 'r' (read only), 'w' (write; default) or 'a'
        (append).
        Ignored if file is a file object.
    path : str, optional
        Path within HDF5-file or group to save data to.
        Defaults to root ('/').
    track_times : bool, optional
        If set to *True* (default), repeated hickling will produce different
        files.
    kwargs : keyword arguments
        Additional keyword arguments that must be provided to the
        :meth:`~h5py._hl.group.Group.create_dataset` method.

    """

    # Make sure that file is not closed unless modified
    # This is to avoid trying to close a file that was never opened
    close_flag = False

    try:
        # Open the file
        h5f, path, close_flag = file_opener(file_obj, path, mode, track_times)

        # Log which version of python was used to generate the hickle file
        pv = sys.version_info
        py_ver = "%i.%i.%i" % (pv[0], pv[1], pv[2])

        # Try to create the root group
        try:
            h_root_group = h5f.create_group(path)

        # If that is not possible, check if it is empty
        except ValueError as error:
            # Raise error if this group is not empty
            if len(h5f[path]):
                raise error
            else:
                h_root_group = h5f.get(path)

        h_root_group.attrs["HICKLE_VERSION"] = __version__
        h_root_group.attrs["HICKLE_PYTHON_VERSION"] = py_ver

        _dump(py_obj, h_root_group, **kwargs)
    finally:
        # Close the file if requested.
        # Closing a file twice will not cause any problems
        if close_flag:
            h5f.file.close()


def create_dataset_lookup(py_obj):
    """ What type of object are we trying to hickle?  This is a python
    dictionary based equivalent of a case statement.  It returns the correct
    helper function for a given data type.

    Args:
        py_obj: python object to look-up what function to use to dump to disk

    Returns:
        match: function that should be used to dump data to a new dataset
        base_type: the base type of the data that will be dumped
    """

    # Obtain the MRO of this object
    mro_list = py_obj.__class__.mro()

    # Create a type_map
    type_map = map(types_dict.get, mro_list)

    # Loop over the entire type_map until something else than None is found
    for type_item in type_map:
        if type_item is not None:
            return(type_item)


def create_hkl_dataset(py_obj, h_group, call_id=None, **kwargs):
    """ Create a dataset within the hickle HDF5 file

    Args:
        py_obj: python object to dump.
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the
            iterable.

    """
    # lookup dataset creator type based on python object type
    create_dataset, base_type = create_dataset_lookup(py_obj)

    # Set the name of this dataset
    name = 'data%s' % ("_%i" % (call_id) if call_id is not None else '')

    # do the creation
    h_subgroup = create_dataset(py_obj, h_group, name, **kwargs)
    h_subgroup.attrs['base_type'] = base_type
    if base_type != b'pickle':
        h_subgroup.attrs['type'] = np.array(pickle.dumps(py_obj.__class__))


def create_hkl_group(py_obj, h_group, call_id=None):
    """ Create a new group within the hickle file

    Args:
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the
            iterable.

    """

    # Set the name of this group
    name = 'data%s' % ("_%i" % (call_id) if call_id is not None else '')

    h_subgroup = h_group.create_group(name)
    h_subgroup.attrs['type'] = np.array(pickle.dumps(py_obj.__class__))
    h_subgroup.attrs['base_type'] = create_dataset_lookup(py_obj)[1]
    return h_subgroup


def create_dict_dataset(py_obj, h_group, name, **kwargs):
    """ Creates a data group for each key in dictionary

    Notes:
        This is a very important function which uses the recursive _dump
        method to build up hierarchical data models stored in the HDF5 file.
        As this is critical to functioning, it is kept in the main hickle.py
        file instead of in the loaders/ directory.

    Args:
        py_obj: python object to dump; should be dictionary
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the
            iterable.
    """
    h_dictgroup = h_group.create_group(name)

    for idx, (key, py_subobj) in enumerate(py_obj.items()):
        # Obtain the string representation of this key
        if isinstance(key, str):
            # Get raw string format of string
            subgroup_key = "%r" % (key)

            # Make sure that the '\\\\' is not in the key, or raise error if so
            if '\\\\' in subgroup_key:
                raise ValueError("Dict item keys containing the '\\\\' string "
                                 "are not supported!")
        else:
            subgroup_key = str(key)

        # Replace any forward slashes with double backslashes
        subgroup_key = subgroup_key.replace('/', '\\\\')
        h_subgroup = h_dictgroup.create_group(subgroup_key)
        h_subgroup.attrs['base_type'] = b'dict_item'

        h_subgroup.attrs['key_base_type'] = type(key).__name__.encode('ascii',
                                                                      'ignore')
        h_subgroup.attrs['key_type'] = np.array(pickle.dumps(key.__class__))

        h_subgroup.attrs['key_idx'] = idx

        _dump(py_subobj, h_subgroup, call_id=None, **kwargs)
    return(h_dictgroup)


# Add create_dict_dataset to types_dict
types_dict[dict] = (create_dict_dataset, b"dict")


###########
# LOADERS #
###########

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
        self.container_base_type = None
        self.name = None
        self.key_type = None
        self.key_base_type = None

    def convert(self):
        """ Convert from PyContainer to python core data type.

        Returns: self, either as a list, tuple, set or dict
                 (or other type specified in lookup.py)
        """

        # If this container is a dict, convert its items properly
        if self.container_base_type == b"dict":
            # Create empty list of items
            items = [[]]*len(self)

            # Loop over all items in the container
            for item in self:
                # Obtain the name of this item
                key = item.name.split('/')[-1].replace('\\\\', '/')

                # Obtain the base type and index of this item's key
                key_base_type = item.key_base_type
                key_idx = item.key_idx

                # If this key has a type that must be converted, do so
                if key_base_type in dict_key_types_dict.keys():
                    to_type_fn = dict_key_types_dict[key_base_type]
                    key = to_type_fn(key)

                # Insert item at the correct index into the list
                items[key_idx] = [key, item[0]]

            # Initialize dict using its true type and return
            return(self.container_type(items))

        # In all other cases, return container
        else:
            # If container has a true type defined, convert to that first
            if self.container_type is not None:
                return(self.container_type(self))

            # If not, return the container itself
            else:
                return(self)


def no_match_load(key):
    """ If no match is made when loading, need to raise an exception
    """
    raise RuntimeError("Cannot load %s data type" % key)


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


def load(file_obj, path='/', safe=True):
    """
    Load the Python object stored in `file_obj` at `path` and return it.

    Parameters
    ----------
    file_obj : file object, str or :obj:`~h5py.Group` object
        File from which to load the object.
        If str, `file_obj` provides the path of the HDF5-file that must be
        used.
        If :obj:`~h5py._hl.group.Group`, the group (or file) in an open
        HDF5-file that must be used.
    path : str, optional
        Path within HDF5-file or group to load data from.
        Defaults to root ('/').
    safe : bool, optional
        Disable automatic depickling of arbitrary python objects.
        DO NOT set this to False unless the file is from a trusted source.
        (See https://docs.python.org/3/library/pickle.html for an explanation)

    Returns
    -------
    py_obj : object
        The unhickled Python object.

    """

    # Make sure that the file is not closed unless modified
    # This is to avoid trying to close a file that was never opened
    close_flag = False

    # Try to read the provided file_obj as a hickle file
    try:
        h5f, path, close_flag = file_opener(file_obj, path)
        h_root_group = h5f.get(path)

        # Define attributes h_root_group must have
        v3_attrs = ['CLASS', 'VERSION', 'PYTHON_VERSION']
        v4_attrs = ['HICKLE_VERSION', 'HICKLE_PYTHON_VERSION']

        # Check if the proper attributes for v3 loading are available
        if all(map(h_root_group.attrs.get, v3_attrs)):
            # Check if group attribute 'CLASS' has value 'hickle
            if(h_root_group.attrs.get('CLASS') != b'hickle'):
                # If not, raise error
                raise AttributeError("HDF5-file attribute 'CLASS' does not "
                                     "have value 'hickle'!")

            # Obtain version with which the file was made
            major_version = int(h_root_group.attrs['VERSION'][0])
            assert major_version == 3

            # Load file
            from hickle import legacy_v3
            warnings.warn("Input argument 'fileobj' appears to be a file made "
                          "with hickle v3. Using legacy load...")
            return(legacy_v3.load(file_obj, path, safe))

        # Else, check if the proper attributes for v4 loading are available
        elif all(map(h_root_group.attrs.get, v4_attrs)):
            # Load file
            py_container = PyContainer()
            py_container = _load(py_container, h_root_group['data'])
            return(py_container[0])

        # Else, raise error
        else:
            pass

    # If this fails, raise error and provide user with caught error message
    except Exception as error:
        raise ValueError("Provided argument 'fileobj' does not appear to be a "
                         "valid hickle file! (%s)" % (error))
    finally:
        # Close the file if requested.
        # Closing a file twice will not cause any problems
        if close_flag:
            h5f.file.close()


def load_dataset(h_node):
    """ Load a dataset, converting into its correct python type

    Args:
        h_node (h5py dataset): h5py dataset object to read

    Returns:
        data: reconstructed python object from loaded data
    """
    py_type, base_type = get_type(h_node)
    load_loader(py_type)

    load_fn = load_dataset_lookup(base_type)
    data = load_fn(h_node)

    # If data is not py_type yet, convert to it (unless it is pickle)
    if base_type != b'pickle' and type(data) != py_type:
        data = py_type(data)
    return data


def _load(py_container, h_group):
    """ Load a hickle file

    Recursive funnction to load hdf5 data into a PyContainer()

    Args:
        py_container (PyContainer): Python container to load data into
        h_group (h5 group or dataset): h5py object, group or dataset, to spider
            and load all datasets.
    """

    # Either a file, group, or dataset
    if isinstance(h_group, h5._hl.group.Group):

        py_subcontainer = PyContainer()
        py_subcontainer.container_base_type = bytes(h_group.attrs['base_type'])

        py_subcontainer.name = h_group.name

        if py_subcontainer.container_base_type == b'dict_item':
            py_subcontainer.key_base_type = h_group.attrs['key_base_type']
            py_obj_type = pickle.loads(h_group.attrs['key_type'])
            py_subcontainer.key_type = py_obj_type
            py_subcontainer.key_idx = h_group.attrs['key_idx']
        else:
            py_obj_type = pickle.loads(h_group.attrs['type'])
            py_subcontainer.container_type = py_obj_type

        # Check if we have an unloaded loader for the provided py_obj
        load_loader(py_obj_type)

        if py_subcontainer.container_base_type not in types_not_to_sort:
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
