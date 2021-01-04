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
from .helpers import (
    PyContainer, NotHicklable, nobody_is_my_name, ToDoError
)
from .fileio import ClosedFileError, FileError, file_opener
from .lookup import (
    #hkl_types_dict, hkl_container_dict, load_loader, load_legacy_loader ,
    create_pickled_dataset, load_nothing, fix_lambda_obj_type,ReferenceManager,
    LoaderManager,link_dtype
)


# All declaration
__all__ = ['dump', 'load']


# %% FUNCTION DEFINITIONS


###########
# DUMPERS #
###########

def _dump(py_obj, h_group, name, memo, loader,attrs={} , **kwargs):
    """ Dump a python object to a group within an HDF5 file.

    This function is called recursively by the main dump() function.

    Parameters:
    -----------
        py_obj: python object to dump.
        h_group (h5.File.group): group to dump data into.
        name (bytes): name of resultin hdf5 group or dataset 
        memo (ReferenceManager): the ReferenceManager object
            responsible for handling all object and type memoisation
            related issues
        attrs (dict): addtional attributes to be stored along with the
            resulting hdf5 group or hdf5 dataset
        kwargs (dict): keyword arguments to be passed to create_dataset
            function
    """

    py_obj_id = id(py_obj)
    py_obj_ref = memo.get(py_obj_id,None)
    if py_obj_ref is not None:
        # reference data sets do not have any base_type and no py_obj_type set
        # as they can be distinguished from pickled data due to their dtype 
        # of type ref_dtype and thus load implicitly will be assigned b'!node-reference!'
        # base_type and hickle.lookup.NodeReference as their py_obj_type
        h_link = h_group.create_dataset(name,data = py_obj_ref[0].ref,dtype = link_dtype)
        h_link.attrs.update(attrs)
        return

    # Check if we have a unloaded loader for the provided py_obj and 
    # retrive the most apropriate method for creating the corresponding
    # representation within HDF5 file
    py_obj_type, (create_dataset, base_type,memoise) = loader.load_loader(py_obj.__class__)
    try:
        h_node,h_subitems = create_dataset(py_obj, h_group, name, **kwargs)
    except NotHicklable:
        h_node,h_subitems = create_pickled_dataset(py_obj, h_group, name, reason = str(NotHicklable), **kwargs)
    else:
        # store base_type and type unless py_obj had to be picled by create_pickled_dataset
        memo.store_type(h_node,py_obj_type,base_type,**kwargs)

    # add addtional attributes and set 'base_type' and 'type'
    # attributes accordingly
    h_node.attrs.update((name,attr) for name,attr in attrs.items() if name != 'type' )

    # ask pickle to try to store
    # if h_node shall be memoised for representing multiple references
    # to the same py_obj instance in the hdf5 file store h_node
    # in the memo dictionary. Store py_obj along with h_node to ensure
    # py_obj_id which represents the memory address of py_obj referrs
    # to py_obj until the whole structure is stored within hickle file.
    if memoise:
        memo[py_obj_id] = (h_node,py_obj)

    # loop through list of all subitems and recursively dump them
    # to HDF5 file
    for h_subname,py_subobj,h_subattrs,sub_kwargs in h_subitems:
        _dump(py_subobj,h_node,h_subname,memo,loader,h_subattrs,**sub_kwargs)


def dump(py_obj, file_obj, mode='w', path='/',*,filename = None,options = {},**kwargs):
    """
    Write a hickled representation of `py_obj` to the provided `file_obj`.

    Parameters
    ----------
    py_obj : object
        Python object to hickle to HDF5.

    file_obj : file object, str, pathlib.Path or :obj:`~h5py.Group` object
        File in which to store the object.
        If str or pathlib.Path, `file_obj` provides the path of the HDF5-file
        that must be used.
        If :file_obj:`~h5py.Group`, the group (or file) in an open
        HDF5-file that must be used.
        If :file_obj: `file`, `file like`, the file handle of the file to
        dump :py_obj: to

    mode : str, optional
        Accepted values are 'r' (read only), 'w' (write; default) or 'a'
        (append). 
        Note: A trailing binary mode flag ('b') is ignored

    path : str, optional
        Path within HDF5-file or group to save data to.
        Defaults to root ('/').

    filename : str,  optional
        name of file, file like object of HDF5 file. 
        Ignored if file_obj is a path string, pathlib.Path object or
        represents an accesible h5py.File, h5py.Group or h5py.Dataset 

    options (dict):
        Each entry in this dict modifies how hickle dumps data to file.
        For example 
            { compact_expand = True }
        would enforce use of compact_expand loader on all classes
        registered with this kind of loader.
            { compact_expand = False }
        would disable compact_expand loader for dumped data even if
        globally turned on. More options may follow.

    kwargs : keyword arguments
        Additional keyword arguments that must be provided to the
        :meth:`~h5py.Group.create_dataset` method.

    Raises:
    -------
        CloseFileError:
            If passed h5py.File, h5py.Group or h5py.Dataset object is not
            accessible which in most cases indicate that unterlying HDF5
            was closed or if file or file like object has already been 
            closed.

        FileError
            If passed file or file like object is not opened for reading or
            in addtion for writing in case mode corresponds to any
            of 'w', 'w+', 'x', 'x+' or a.

        ValueError:
            If anything else than str, bytes or None specified for filename
             

    """

    # Make sure that file is not closed unless modified
    # This is to avoid trying to close a file that was never opened
    close_flag = False

    # Open the file
    h5f, path, close_flag = file_opener(file_obj, path, mode,filename)
    try:

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

        with LoaderManager.create_manager(h_root_group,False,options) as loader:
            with ReferenceManager.create_manager(h_root_group) as memo:
                _dump(py_obj, h_root_group,'data', memo ,loader,**kwargs)
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
        

def no_match_load(key,*args,**kwargs):     # pragma: no cover
    """ 
    If no match is made when loading dataset , need to raise an exception
    """
    raise RuntimeError("Cannot load %s data type" % key)

def load(file_obj, path='/', safe=True, filename = None):
    """
    Load the Python object stored in `file_obj` at `path` and return it.

    Parameters
    ----------

    file_obj : file object, str, pathlib.Path or :obj:`~h5py.Group` object
        File in which to store the object.
        If str or pathlib.Path, `file_obj` provides the path of the HDF5-file
        that must be used.
        If :file_obj:`~h5py.Group`, the group (or file) in an open
        HDF5-file that must be used.
        If :file_obj: `file`, `file like`, the file handle of the file to
        load :py_obj: from.

    path : str, optional
        Path within HDF5-file or group to load data from.
        Defaults to root ('/').

    safe : bool, optional
        Disable automatic depickling of arbitrary python objects.
        DO NOT set this to False unless the file is from a trusted source.
        (See https://docs.python.org/3/library/pickle.html for an explanation)

    filename : str,  optional
        name of file, file like object of HDF5 file. 
        Ignored if file_obj is a path string, pathlib.Path object or
        represents an accesible h5py.File, h5py.Group or h5py.Dataset 

    Returns
    -------
    py_obj : object
        The unhickled Python object.

    Raises:
    -------
        CloseFileError:
            If passed h5py.File, h5py.Group or h5py.Dataset object is not
            accessible which in most cases indicate that unterlying HDF5
            was closed or if file or file like object has already been 
            closed.

        FileError
            If passed file or file like object is not opened for reading or
            in addtion for writing in case mode corresponds to any
            of 'w', 'w+', 'x', 'x+' or a.

        ValueError:
            If anything else than str, bytes or None specified for filename
    """

    # Make sure that the file is not closed unless modified
    # This is to avoid trying to close a file that was never opened
    close_flag = False

    # Try to read the provided file_obj as a hickle file
    h5f, path, close_flag = file_opener(file_obj, path, 'r')
    try:
        h_root_group = h5f.get(path,None) # Soley used by v4
        if not isinstance(h_root_group,h5.Group):
            raise FileError("file '{}': path '{}' not exising".format(h5f.filename,path))

        # Define attributes h_root_group must have
        v3_attrs = ['CLASS', 'VERSION', 'PYTHON_VERSION']
        v4_attrs = ['HICKLE_VERSION', 'HICKLE_PYTHON_VERSION']

        # Check if the proper attributes for v3 loading are available
        if all(map(h5f.attrs.get, v3_attrs)):
            # Check if group attribute 'CLASS' has value 'hickle
            if(h5f.attrs['CLASS'] not in ( b'hickle','hickle')):  # pragma: no cover
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
                with LoaderManager.create_manager(h_root_group,True) as loader:
                    with ReferenceManager.create_manager(h_root_group,fix_lambda_obj_type) as memo:
                        _load(py_container, 'data',h_root_group['data'],memo,loader) #load_loader = load_legacy_loader)
                return py_container.convert()
            # 4.1.x file and newer
            with LoaderManager.create_manager( h_root_group,False) as loader:
                with ReferenceManager.create_manager(h_root_group,pickle_loads) as memo:
                    _load(py_container, 'data',h_root_group['data'],memo,loader) #load_loader = load_loader)
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



def _load(py_container, h_name, h_node,memo,loader): #load_loader = load_loader):
    """ Load a hickle file

    Recursive funnction to load hdf5 data into a PyContainer()

    Args:
        py_container (PyContainer): Python container to load data into
        h_name (string): the name of the resulting h5py object group or dataset
        h_node (h5 group or dataset): h5py object, group or dataset, to spider
            and load all datasets.
        memo (ReferenceManager): the ReferenceManager object
            responsible for handling all object and type memoisation
            related issues
        load_loader (FunctionType,MethodType): defaults to lookup.load_loader and
            will be switched to load_legacy_loader if file to be loaded was
            created by hickle 4.0.x version
    """

    # if h_node has already been loaded cause a reference to it was encountered earlier
    # direcctly append it to its parent container and return
    node_ref = memo.get(h_node.id,h_node)
    if node_ref is not h_node:
        py_container.append(h_name,node_ref,h_node.attrs)
        return

    # load the type information of node.
    py_obj_type,base_type,is_container = memo.resolve_type(h_node)
    py_obj_type,(_,_,memoise) = loader.load_loader(py_obj_type)
    
    if is_container:
        # Either a h5py.Group representing the structure of complex objects or
        # a h5py.Dataset representing a h5py.Reference to the node of an object
        # referred to from multiple places within the objet structure to be dumped 

        py_container_class = loader.hkl_container_dict.get(base_type,NoMatchContainer)
        py_subcontainer = py_container_class(h_node.attrs,base_type,py_obj_type)
    
        # NOTE: Sorting of container items according to their key Name is
        #       to be handled by container class provided by loader only
        #       as loader has all the knowledge required to properly decide
        #       if sort is necessary and how to sort and at what stage to sort 
        for h_key,h_subnode in py_subcontainer.filter(h_node):
            _load(py_subcontainer, h_key, h_subnode, memo ,loader) # load_loader)

        # finalize subitem
        sub_data = py_subcontainer.convert()
        py_container.append(h_name,sub_data,h_node.attrs)
    else:
        # must be a dataset load it and append to parent container
        load_fn = loader.hkl_types_dict.get(base_type, no_match_load)
        sub_data = load_fn(h_node,base_type,py_obj_type)
        py_container.append(h_name,sub_data,h_node.attrs)
    # store loaded object for properly restoring addtional references to it
    if memoise:
        memo[h_node.id] = sub_data

