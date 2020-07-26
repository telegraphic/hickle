# encoding: utf-8
"""
# hickle.py

Created by Danny Price 2016-02-03.

Hickle is an HDF5 based clone of Pickle. Instead of serializing to a pickle
file, Hickle dumps to an HDF5 file. It is designed to be as similar to pickle
in usage as possible, providing a load() and dump() function.

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
import os.path as os_path
from pathlib import Path
import sys
import warnings
import types
import functools as ft

# Package imports
import dill as pickle
import h5py as h5
import numpy as np

# hickle imports
from hickle import __version__
from .helpers import PyContainer, NotHicklable, nobody_is_my_name
from .lookup import (
    hkl_types_dict, hkl_container_dict, load_loader, load_legacy_loader ,
    create_pickled_dataset, load_nothing, fix_lambda_obj_type
)


# All declaration
__all__ = ['dump', 'load']


# %% CLASS DEFINITIONS
##################
# Error handling #
##################

class FileError(Exception):
    """ An exception raised if the file is fishy """


class ClosedFileError(Exception):
    """ An exception raised if the file is fishy """


class ToDoError(Exception):     # pragma: no cover
    """ An exception raised for non-implemented functionality"""
    def __str__(self):
        return "Error: this functionality hasn't been implemented yet."

class SerializedWarning(UserWarning):
    """ An object type was not understood
    The data will be serialized using pickle.
    """

# %% FUNCTION DEFINITIONS
def file_opener(f, path, mode='r'):
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
        If :obj:`~h5py.Group`, the group (or file) in an open
        HDF5-file that must be used.
    path : str
        Path within HDF5-file or group to dump to/load from.
    mode : str, optional
        Accepted values are 'r' (read only), 'w' (write; default) or 'a'
        (append).
        Ignored if file is a file object.

    """

    # Assume that we will have to close the file after dump or load
    close_flag = True

    # Make sure that the given path always starts with '/'
    if not path.startswith('/'):
        path = "/%s" % path

    # Were we handed a file object or just a file name string?
    if isinstance(f, (io.TextIOWrapper, io.BufferedWriter)):
        filename, mode = f.name, f.mode
        f.close()
        mode = mode.replace('b', '')
        h5f = h5.File(filename, mode)
    elif isinstance(f, (str, Path)):
        filename = f
        h5f = h5.File(filename, mode)
    elif isinstance(f, h5.Group):
        try:
            filename = f.file.filename
        except ValueError:
            raise ClosedFileError("HDF5 file has been closed. Please pass "
                                  "either a filename string, a file object, or"
                                  "an open HDF5-file")
        path = ''.join([f.name, path])
        h5f = f.file

        if path.endswith('/'):
            path = path[:-1]

        # Since this file was already open, do not close the file afterward
        close_flag = False

    else:
        print(f.__class__)
        raise FileError("Cannot open file. Please pass either a filename "
                        "string, a file object, or a h5py.File")

    return h5f, path, close_flag


###########
# DUMPERS #
###########

def _dump(py_obj, h_group, name, attrs={} , **kwargs):
    """ Dump a python object to a group within an HDF5 file.

    This function is called recursively by the main dump() function.

    Args:
        py_obj: python object to dump.
        h_group (h5.File.group): group to dump data into.
        name (bytes): name of resultin hdf5 group or dataset 
    """

    # Check if we have a unloaded loader for the provided py_obj and 
    # retrive the most apropriate method for crating the corresponding
    # representation within HDF5 file
    if isinstance(
         py_obj,
         (types.FunctionType, types.BuiltinFunctionType, types.MethodType, types.BuiltinMethodType, type)
    ):
        py_obj_type,create_dataset,base_type = object,create_pickled_dataset,b'pickle'
    else:
        py_obj_type, (create_dataset, base_type) = load_loader(py_obj.__class__)
    try:
        h_node,h_subitems = create_dataset(py_obj, h_group, name, **kwargs)

        # loop through list of all subitems and recursively dump them
        # to HDF5 file
        for h_subname,py_subobj,h_subattrs,sub_kwargs in h_subitems:
            _dump(py_subobj,h_node,h_subname,h_subattrs,**sub_kwargs)
        # add addtional attributes and set 'base_type' and 'type'
        # attributes accordingly
        h_node.attrs.update(attrs)

        # only explicitly store base_type and type if not dumped by
        # create_pickled_dataset
        if create_dataset is not create_pickled_dataset:
            h_node.attrs['base_type'] = base_type
            h_node.attrs['type'] = np.array(pickle.dumps(py_obj_type))
        return
    except NotHicklable:

        # ask pickle to try to store
        h_node,h_subitems = create_pickled_dataset(py_obj, h_group, name, reason = str(NotHicklable), **kwargs)
        h_node.attrs.update(attrs)


def dump(py_obj, file_obj, mode='w', path='/', **kwargs):
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
        If :obj:`~h5py.Group`, the group (or file) in an open
        HDF5-file that must be used.
    mode : str, optional
        Accepted values are 'r' (read only), 'w' (write; default) or 'a'
        (append).
        Ignored if file is a file object.
    path : str, optional
        Path within HDF5-file or group to save data to.
        Defaults to root ('/').
    kwargs : keyword arguments
        Additional keyword arguments that must be provided to the
        :meth:`~h5py.Group.create_dataset` method.

    """

    # Make sure that file is not closed unless modified
    # This is to avoid trying to close a file that was never opened
    close_flag = False

    try:
        # Open the file
        h5f, path, close_flag = file_opener(file_obj, path, mode)

        # Log which version of python was used to generate the hickle file
        pv = sys.version_info
        py_ver = "%i.%i.%i" % (pv[0], pv[1], pv[2])

        h_root_group = h5f.get(path,None)
        if h_root_group is None:
            h_root_group = h5f.create_group(path)
        elif h_root_group.items():
            raise ValueError("Unable to create group (name already exists)")

        h_root_group.attrs["HICKLE_VERSION"] = __version__
        h_root_group.attrs["HICKLE_PYTHON_VERSION"] = py_ver

        _dump(py_obj, h_root_group,'data', **kwargs)
    finally:
        # Close the file if requested.
        # Closing a file twice will not cause any problems
        if close_flag:
            h5f.close()

###########
# LOADERS #
###########

class RootContainer(PyContainer):
    """
    PyContainer representing the whole HDF5 file
    """

    __slots__ = ()
    def convert(self):
        return self._content[0]


class NoMatchContainer(PyContainer): # pragma: no cover
    """
    PyContainer used by load when no appropriate container
    could be found for specified base_type. 
    """

    __slots__ = ()

    def __init__(self,h5_attrs, base_type, object_type): # pragma: no cover
        raise RuntimeError("Cannot load container proxy for %s data type " % base_type)
        

def no_match_load(key):     # pragma: no cover
    """ 
    If no match is made when loading dataset , need to raise an exception
    """
    raise RuntimeError("Cannot load %s data type" % key)

def load(file_obj, path='/', safe=True):
    """
    Load the Python object stored in `file_obj` at `path` and return it.

    Parameters
    ----------
    file_obj : file object, str or :obj:`~h5py.Group` object
        File from which to load the object.
        If str, `file_obj` provides the path of the HDF5-file that must be
        used.
        If :obj:`~h5py.Group`, the group (or file) in an open
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
        h5f, path, close_flag = file_opener(file_obj, path, 'r')
        h_root_group = h5f.get(path)   # Solely used by v4

        # Define attributes h_root_group must have
        v3_attrs = ['CLASS', 'VERSION', 'PYTHON_VERSION']
        v4_attrs = ['HICKLE_VERSION', 'HICKLE_PYTHON_VERSION']

        # Check if the proper attributes for v3 loading are available
        if all(map(h5f.attrs.get, v3_attrs)):
            # Check if group attribute 'CLASS' has value 'hickle
            if(h5f.attrs['CLASS'] != b'hickle'):  # pragma: no cover
                # If not, raise error
                raise AttributeError("HDF5-file attribute 'CLASS' does not "
                                     "have value 'hickle'!")

            # Obtain version with which the file was made
            try:
                major_version = int(h5f.attrs['VERSION'][0])

            # If this cannot be done, then this is not a v3 file
            except Exception:  # pragma: no cover
                raise Exception("This file does not appear to be a hickle v3 "
                                "file.")

            # Else, if the major version is not 3, it is not a v3 file either
            else:
                if(major_version != 3):  # pragma: no cover
                    raise Exception("This file does not appear to be a hickle "
                                    "v3 file.")

            # Load file
            from hickle import legacy_v3
            warnings.warn("Input argument 'file_obj' appears to be a file made"
                          " with hickle v3. Using legacy load...")
            return(legacy_v3.load(file_obj, path, safe))

        # Else, check if the proper attributes for v4 loading are available
        if all(map(h_root_group.attrs.get, v4_attrs)):
            # Load file
            py_container = RootContainer(h_root_group.attrs,b'document_root',RootContainer)
            pickle_loads = pickle.loads
            hickle_version = h_root_group.attrs["HICKLE_VERSION"].split('.')
            if int(hickle_version[0]) == 4 and int(hickle_version[1]) < 1:
                # hickle 4.0.x file activate if legacy load fixes for 4.0.x
                # eg. pickle of versions < 3.8 do not prevent dumping of lambda functions
                # eventhough stated otherwise in documentation. Activate workarrounds
                # just in case issues arrise. Especially as corresponding lambdas in
                # load_numpy are not needed anymore and thus have been removed.
                pickle_loads = fix_lambda_obj_type
                _load(py_container, 'data',h_root_group['data'],pickle_loads = fix_lambda_obj_type,load_loader = load_legacy_loader)
                return py_container.convert()
            # 4.1.x file and newer
            _load(py_container, 'data',h_root_group['data'],pickle_loads = pickle.loads,load_loader = load_loader)
            return py_container.convert()

        # Else, raise error
        raise FileError("HDF5-file does not have the proper attributes!")

    # If this fails, raise error and provide user with caught error message
    except Exception as error:
        raise ValueError("Provided argument 'file_obj' does not appear to be a valid hickle file! (%s)" % (error),error) from error
    finally:
        # Close the file if requested.
        # Closing a file twice will not cause any problems
        if close_flag:
            h5f.close()



def _load(py_container, h_name, h_node,pickle_loads=pickle.loads,load_loader = load_loader):
    """ Load a hickle file

    Recursive funnction to load hdf5 data into a PyContainer()

    Args:
        py_container (PyContainer): Python container to load data into
        h_name (string): the name of the resulting h5py object group or dataset
        h_node (h5 group or dataset): h5py object, group or dataset, to spider
            and load all datasets.
        pickle_loads (FunctionType,MethodType): defaults to pickle.loads and will
            be switched to fix_lambda_obj_type if file to be loaded was created by
            hickle 4.0.x version
        load_loader (FunctionType,MethodType): defaults to lookup.load_loader and
            will be switched to load_legacy_loader if file to be loaded was
            created by hickle 4.0.x version
    """

    # load base_type of node. if not set assume that it contains
    # pickled object data to be restored through load_pickled_data or
    # PickledContainer object in case of group.
    base_type = h_node.attrs.get('base_type',b'pickle')
    if base_type == b'pickle':
        # pickled dataset or group assume its object_type to be object
        # as true object type is anyway handled by load_pickled_data or
        # PickledContainer
        py_obj_type = object
    else:
        # extract object_type and ensure loader beeing able to handle is loaded
        # loading is controlled through base_type, object_type is just required
        # to allow load_fn or py_subcontainer to properly restore and cast
        # py_obj to proper object type
        py_obj_type = pickle_loads(h_node.attrs.get('type',None))
        py_obj_type,_ = load_loader(py_obj_type)
    
    # Either a file, group, or dataset
    if isinstance(h_node, h5.Group):

        py_container_class = hkl_container_dict.get(base_type,NoMatchContainer)
        py_subcontainer = py_container_class(h_node.attrs,base_type,py_obj_type)
    
        # NOTE: Sorting of container items according to their key Name is
        #       to be handled by container class provided by loader only
        #       as loader has all the knowledge required to properly decide
        #       if sort is necessary and how to sort and at what stage to sort 
        for h_key,h_subnode in py_subcontainer.filter(h_node.items()):
            _load(py_subcontainer, h_key, h_subnode, pickle_loads, load_loader)

        # finalize subitem and append to parent container.
        sub_data = py_subcontainer.convert()
        py_container.append(h_name,sub_data,h_node.attrs)

    else:
        # must be a dataset load it and append to parent container
        load_fn = hkl_types_dict.get(base_type, no_match_load)
        data = load_fn(h_node,base_type,py_obj_type)
        py_container.append(h_name,data,h_node.attrs)

