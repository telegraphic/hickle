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

3) pickle.dumps() and pickle.loads() functions can be mimicked by passing
a BytesIO type to hickle.dump() or hickle.load() function and setting the
filename parameter to a non empty string.

    hicklestring = BytesIO()
    hickle.dump(my_data,hicklestring,mode='w',filename='<string>')

    loaded_data = hickle.load(hicklestring,mode='r',filename='<string>')

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
    LoaderManager, RecoverGroupContainer, recover_custom_dataset
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

    Parameters
    ----------
    py_obj (object):
         python object to dump.

    h_group (h5.File.group):
         group to dump data into.

    name (str):
         name of resulting hdf5 group or dataset 

    memo (ReferenceManager):
        the ReferenceManager object responsible for handling all
        object and type memoisation related issues

    attrs (dict):
        additional attributes to be stored along with the resulting
        hdf5 group or hdf5 dataset

    kwargs (dict):
        keyword arguments to be passed to create_dataset function
    """

    py_obj_id = id(py_obj)
    py_obj_ref = memo.get(py_obj_id,None)
    if py_obj_ref is not None:

        # py_object already dumped to hdf5 file store a refrence to it instead
        # instead of dumping it again.
        #
        # Note: reference dataset share their base_type and py_obj_type with the
        #       referenced h5py.Group or h5py.Dataset. On load their h5py.ref_dtype type 
        #       dtype is used to distinguish them from datasets hosting pickled data.
        h_link = h_group.create_dataset(name,data = py_obj_ref[0].ref,dtype = h5.ref_dtype)
        h_link.attrs.update(attrs)
        return

    # Check if loader has already been loaded for the provided py_obj and 
    # retrieve the most appropriate method for creating the corresponding
    # representation within HDF5 file
    py_obj_type, (create_dataset, base_type,memoise) = loader.load_loader(py_obj.__class__)
    try:
        h_node,h_subitems = create_dataset(py_obj, h_group, name, **kwargs)
    except NotHicklable:
        h_node,h_subitems = create_pickled_dataset(py_obj, h_group, name, reason = str(NotHicklable), **kwargs)
    else:
        # store base_type and type unless py_obj had to be pickled by create_pickled_dataset
        memo.store_type(h_node,py_obj_type,base_type,**kwargs)

    # add additional attributes and prevent modification of 'type' attribute
    h_node.attrs.update((name,attr) for name,attr in attrs.items() if name != 'type' )

    # if py_object shall be memoised to properly represent multiple references
    # to it in HDF5 file store it along with created h_node in the memo dictionary.
    # remembering the py_object along with the h_node ensures that py_object_id
    # which represents the memory address of py_obj refers to py_obj until the
    # whole structure is stored within hickle file.
    if memoise:
        memo[py_obj_id] = (h_node,py_obj)

    # loop through list of all sub items and recursively dump them
    # to HDF5 file
    for h_subname,py_subobj,h_subattrs,sub_kwargs in h_subitems:
        _dump(py_subobj,h_node,h_subname,memo,loader,h_subattrs,**sub_kwargs)


def dump(py_obj, file_obj, mode='w', path='/',*,filename = None,options = {},**kwargs):
    """
    Write a hickled representation of `py_obj` to the provided `file_obj`.

    Parameters
    ----------
    py_obj (object):
        Python object to hickle to HDF5.

    file_obj (file, file-like, h5py.File, str, (file,str),{'file':file,'name':str} ):
        File to open for dumping or loading purposes.
        str:
            the path of the HDF5-file that must be used.
        ~h5py.Group:
             the group (or file) in an open HDF5-file that must be used.
        file, file-like: 
            file or like object which provides `read`, `seek`, `tell` and write methods
        tuple:
            two element tuple with the first being the file or file like object
            to dump to and the second the filename to be used instead of 'filename'
            parameter
        dict:
            dictionary with 'file' and 'name' items

    mode (str): optional
        string indicating how the file shall be opened. For details see Python `open`.
        
        Note: The 'b' flag is optional as all files are and have to be opened in
            binary mode.

    path (str): optional
        Path within HDF5-file or group to dump to/load from.
    
    filename (str): optional
        The name of the file. Ignored when f is `str` or `h5py.File` object.

    options (dict): optional
        Each entry in this dict modifies how hickle dumps data to file.
        For example 
            { custom = True }
        would enforce use of custom loaders on all classes
        registered with this kind of loader.
            { custom = False }
        would disable custom loaders for dumped data even if
        globally turned on. More options may follow.

    kwargs : keyword arguments
        Additional keyword arguments that must be provided to the
        :meth:`~h5py.Group.create_dataset` method. For example compression=True

    Raises
    ------
    CloseFileError:
        If passed h5py.File, h5py.Group or h5py.Dataset object is not
        accessible. This in most cases indicates that underlying HDF5
        was closed or if file or file or file-like object has already been 
        closed.

    FileError
        If passed file or file-like object is not opened for reading or
        in addition for writing in case mode corresponds to any
        of 'w', 'w+', 'x', 'x+' or a.

    ValueError:
        If anything else than str, bytes or None specified for filename
        or for mode is anything else specified than 'w','w+','x','x+','r','r+','a'
        or contains any optional open flag other than 'b'
    """

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
        # Close the h5py.File if it was opened by hickle.
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


def load(file_obj, path='/', safe=True, filename = None):
    """
    Load the Python object stored in `file_obj` at `path` and return it.

    Parameters
    ----------

    file_obj (file, file-like, h5py.File, str, (file,str),{'file':file,'name':str} ):
        File to open for dumping or loading purposes.
        str:
            the path of the HDF5-file that must be used.
        ~h5py.Group:
             the group (or file) in an open HDF5-file that must be used.
        file, file-like: 
            file or like object which provides `read`, `seek`, `tell` and write methods
        tuple:
            two element tuple with the first being the file or file like object
            to dump to and the second the filename to be used instead of 'filename'
            parameter
        dict:
            dictionary with 'file' and 'name' items

    
    path (str): optional
            Path within HDF5-file or group to dump to/load from.

    safe (bool): optional
        Disable automatic depickling of arbitrary python objects.
        DO NOT set this to False unless the file is from a trusted source.
        (See https://docs.python.org/3/library/pickle.html for an explanation)

        Note: ignored when loading hickle 4.x and newer files

        
    filename (str): optional
        The name of the file. Ignored when f is `str` or `h5py.File` object.

    Returns
    -------
    py_obj : object
        The unhickled Python object.

    Raises
    ------
    CloseFileError:
        If passed h5py.File, h5py.Group or h5py.Dataset object is not
        accessible. This in most cases indicates that underlying HDF5
        was closed or if file or file or file-like object has already been 
        closed.

    FileError
        If passed file or file-like object is not opened for reading

    ValueError:
        If anything else than str, bytes or None specified for filename
    """

    # Try to read the provided file_obj as a hickle file
    h5f, path, close_flag = file_opener(file_obj, path, 'r')
    try:
        h_root_group = h5f.get(path,None) # only used by v4
        if not isinstance(h_root_group,h5.Group):
            raise FileError("file '{}': path '{}' not existing".format(h5f.filename,path))

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
            if int(hickle_version[0]) == 4:
                # hickle 4.x file activate if legacy load fixes for 4.x
                # eg. pickle of versions < 3.8 do not prevent dumping of lambda functions
                # even though stated otherwise in documentation. Activate workarounds
                # just in case issues arise. Especially as corresponding lambdas in
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

    Recursive function to load hdf5 data into a PyContainer()

    Parameters
    ----------
    py_container (PyContainer):
        Python container to load data into

    h_name (str):
        the name of the resulting h5py.Group or h5py.Dataset

    h_node (h5py.Group, h5py.Dataset):
        h5py.Group or h5py.Dataset to restore data from.

    memo (ReferenceManager):
        the ReferenceManager object responsible for handling all object
        and type memoisation related issues

    loader (LoaderManager):
        the LoaderManager object managing the loaders required to properly
        restore the content of h_node and append it to py_container.
    """

    # if h_node has already been loaded cause a reference to it was encountered
    # earlier directly append it to its parent container and return
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
        # referred to from multiple places within the object structure on dump 
        # is to be restored.
        # If no appropriate PyContainer is available use RecoverGroupContainer
        # instead to at least recover its contained data

        py_container_class = loader.hkl_container_dict.get(base_type,RecoverGroupContainer)
        py_subcontainer = py_container_class(h_node.attrs,base_type,py_obj_type)
    
        for h_key,h_subnode in py_subcontainer.filter(h_node):
            _load(py_subcontainer, h_key, h_subnode, memo ,loader)

        # finalize sub item
        sub_data = py_subcontainer.convert()
        py_container.append(h_name,sub_data,h_node.attrs)
    else:

        # must be a dataset load it and append to parent container.
        # In case no appropriate loader could be found use recover_custom_dataset
        # instead to at least recover the contained data
        load_fn = loader.hkl_types_dict.get(base_type, recover_custom_dataset)
        sub_data = load_fn(h_node,base_type,py_obj_type)
        py_container.append(h_name,sub_data,h_node.attrs)

    # store loaded object for properly restoring additional references to it
    if memoise:
        memo[h_node.id] = sub_data

