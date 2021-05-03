# encoding: utf-8
"""
#lookup.py

This file manages all the mappings between hickle/HDF5 metadata and python
types.
There are three dictionaries that are populated by the LoaderManager associated
with each file created or loaded:

1) types_dict
Mapping between python types and dataset and group creation functions, e.g.
    types_dict = {
        list: (create_listlike_dataset, 'list'),
        int: (create_python_dtype_dataset, 'int'),
        np.ndarray: (create_np_array_dataset, 'ndarray'),
        }

2) hkl_types_dict
Mapping between hickle metadata and dataset loading functions, e.g.
    hkl_types_dict = {
        'list': load_list_dataset,
        'tuple': load_tuple_dataset
        }

3) hkl_container_dict
Mapping between hickle metadata and group container classes, e.g.
    hkl_contianer_dict = {
        'list': ListLikeContainer,
        'tuple': TupleLikeContainer,
        'dict': DictLikeContainer
    }

## Extending hickle to add support for other classes and types

The process to add new load/dump capabilities is as follows:

1) Create a file called load_[newmodule/newpackage].py
2) In the load_[newmodule/newpackage].py file, define your create_dataset,
   load_dataset functions and PyContainer objects, along with the 
   'class_register' and 'exclude_register' tables. See the other loaders
   in <Python path>/dist_packages/hickle/loaders directory for examples.
   The columns in register_class table correspond to argument list of 
   hickle.lookup.LoaderManager.register_class method. 
   

3) store the load_[newmodule/newpackage].py file in one of the following locations
   <Python path>/dist_packages/hickle/loaders/
       the loaders supported by hickle globally

   <Python path>/dist_packages/hickle_loaders/
       loaders installed during installation of additional single file python modules
       through pip or legacy os installer

   <Python path>/dist_packages/[newpackage]/hickle_loaders/
       loaders provided by [newpackage] installed through pip or legacy os installer

   [MyPackage]/hickle_loaders/
       loaders specific for objects and classes defined by the modules of [MyPackage] package

   basedir([MyModule|MyApplication])/hickle_loaders
       loaders specific for object and classes defined by [MyModule|MyApplication].py file
       not part of a python package

Loader for a single object or class can also be created by calling 
hickle.lookup.register_class prior to calling hickle.dump and hickle.load

1) create a `create_[MyClass]_dataset` function, a `load_[MyClass]_fcn` function and/or
   `[MyClass]Container` class for [MyClass] class to be dumped and loaded by hickle.
   Examples can be found in the loader modules in <Python path>/dist_packages/hickle/loaders
   directory
2) call `hickle.lookup.LoaderManager.register_class` method as follows from your code

   ```
   from hickle.lookup import LoaderManager

   LoaderManager.register_class(
       [MyClass],'<myclass_type>',
       create_[MyClass]_dataset,
       load_[MyClass]_fcnt, # or None if [MyClass] is mapped to h5py.Group only
       [MyClass]Container , # or None if [MyClass] is mapped to h5py.Dataset only
       True, # if False [MyClass] object will be stored explicitly on any occurrence
       'custom', # set to None to enforce unconditional use of loader
   )

   hickle.dump('my_[MyClass]_object,'myfile.hkl','w',options={'custom':True})
   new_[MyClass]_object = hickle.load('myfile.hkl')
   ```

"""


# %% IMPORTS
# Built-in imports
import sys
import warnings
import types
import re
import weakref
import os.path
from importlib.util import find_spec, module_from_spec,spec_from_file_location,spec_from_loader
from importlib import invalidate_caches

# Package imports
import collections
import dill as pickle
import numpy as np
import h5py as h5

# hickle imports
from .helpers import PyContainer,not_dumpable,nobody_is_my_name,no_compression,NotHicklable
from .loaders import optional_loaders, attribute_prefix

if sys.version_info[:2] <= (3,5): # pragma: no cover
    # define ModuleNotFoundError on __builtins__ to ensure below code is working
    setattr(sys.modules['builtins'],'ModuleNotFoundError',getattr(sys.modules['builtins'],'ModuleNotFoundError',ImportError))

# %% GLOBALS

# %% FUNCTION DEFINITIONS


def load_nothing(h_node, base_type , py_obj_type): # pragma: no cover
    """
    loads nothing
    """
    return nobody_is_my_name

def dump_nothing(py_obj, h_group, name, **kwargs): # pragma: no cover
    """
    dumps nothing
    """
    return nobody_is_my_name


# %% CLASS DEFINITIONS

class _DictItem(): # pragma: no cover
    """
    dummy py_obj for dict_item loader
    """

class NodeReference(): # pragma: no cover
    """
    dummy py_obj_type returned by ReferenceManager.get_type
    when encountering dataset of h5py.ref_dtype  which expose 
    no explicit 'type' attribute.
    """

class ReferenceError(Exception): # pragma: no cover
    """
    exception thrown by ReferenceManager
    """

class LookupError(Exception): # pragma: no cover
    """
    exception thrown if type lookup fails
    """

class SerializedWarning(UserWarning): # pragma: no cover
    """ An object type was not understood

    The data will be serialized using pickle.
    """

class PackageImportDropped(UserWarning): # pragma: no cover
    """
    Package or module defining type/class was removed from 
    sys.modules
    """

class MockedLambdaWarning(UserWarning): # pragma: no cover
    """
    In Python >= 3.8 lambda function fails pickle.loads

    faking restoring lambda to keep legacy hickle 4.x files
    loading properly
    """

class DataRecoveredWarning(UserWarning): # pragma: no cover
    """
    Raised when hickle does not find an appropriate loader for
    a specific type and thus has to fall back to '!recover!' loader
    recover_custom_dataset function or RecoverGroupContainer PyContainer
    object
    """

class AttemptRecoverCustom():
    """
    Dummy type indicating that the data of specific py_obj_type listed in the
    hickle_types_table could not be restored. Most likely pickle.loads encountered
    an ImportError/ModuleNotFoundError or AttributeError indicating that the package
    and/or module defining the py_obj_type is not installed or does not anymore 
    provide the definition of py_object_type.

    In this case hickle tries to at least recover the data and corresponding meta
    data stored in h5py.Group or h5py.Dataset attributes and the base_type string
    indicating the loader used to dump the data.  Only in case this attempt fails
    an exception is thrown.
    """

class RecoveredGroup(dict,AttemptRecoverCustom):
    """
    dict type object representing the content and hickle meta data of a h5py.group
    """

    __slots__ = ('attrs',)

    def __init__(self,*args,attrs={},**kwargs):
        super().__init__(*args,**kwargs)
        self.attrs={name:value for name,value in attrs.items() if name not in {'type'}}

class RecoveredDataset(np.ndarray,AttemptRecoverCustom):
    """
    numpy.ndarray type object representing the content and hickle meta data  of a
    h5py.dataset
    """

    __slots__ = ('attrs',)

    def __new__(cls,input_array,dtype=None,attrs={}):
        array_copy = np.array(input_array,dtype=dtype)
        obj = super().__new__(
            cls,
            shape = array_copy.shape,
            dtype = array_copy.dtype,
            buffer=array_copy,
            offset=0,
            strides=array_copy.strides,
            order = 'C' if array_copy.flags.c_contiguous else 'F'
        )
        obj.attrs = {name:value for name,value in attrs.items() if name not in {'type'}}
        return obj
    
    def __array_finalize__(self,obj):
        if obj is not None:
            self.attrs = getattr(obj,'attrs',{})

class ManagerMeta(type):
    """
    Metaclass for all manager classes derived from the BaseManager class.
    Ensures that the __managers__ class attribute of each immediate child
    class of BaseManager is initialized to a dictionary and ensures that
    it can not be overwritten by any grandchild class and down. The
    __managers__ attribute declared by any grandchild shadowing the
    __managers__ attribute of any of its ancestors is dropped without any
    further notice.
    """

    def __new__(cls, name, bases, namespace, **kwords):
        for _ in ( True for base in bases if isinstance(getattr(base,'__managers__',None),dict) ):
            namespace.pop('__managers__',None)
            break
        else:
            if not isinstance(namespace.get('__managers__',None),dict):
                namespace['__managers__'] = dict() if bases and not object in bases else None
        return super().__new__(cls,name,bases,namespace,**kwords)
                
class BaseManager(metaclass = ManagerMeta):
    """
    Base class providing basic management of simultaneously open managers.
    Must be subclassed.
    """
    __slots__ = ('__weakref__',)

    __managers__ = None

    @classmethod
    def _drop_manager(cls, fileid):
        """
        finalizer callback to properly remove a <manager> object from
        the <manager>.__managers__ structure when corresponding file is
        closed or <manager> object is garbage collected

        Parameters
        ----------
        cls (BaseManager):
            the child <manager> class for which to drop the hdf5 file instance
            referenced to by the specified file id

        fileid (h5py.FileId):
            Id identifying the hdf5 file the ReferenceManager was created for

        """
        try:
            manager = cls.__managers__.get(fileid,None)
            if manager is None:
                return
            cls.__managers__.pop(fileid,None)
        except: # pragma: no cover
            # only triggered in case of race
            pass

    @classmethod
    def create_manager(cls, h_node, create_entry):
        """
        Check whether an instance already exists for the file the h_node belongs to and
        call create_entry if no entry yet exists.

        Parameters
        ----------
        cls:
            the manager class to create a new instance for

        h_node (h5py.File, h5py.Group, h5py.Dataset):
            the h5py node or its h_root_group or file to create a new ReferenceManager
            object for.

        create_entry (callable):
            function or method which returns a new table entry. A table entry
            is a tuple or list with contains as its first item the newly
            created <manager> object. It may include further items specific to
            the actual BaseManager subclass.
                
        Raises
        ------
        LookupError:
            if manager has already been created for h_node or its h_root_group 
                
        """
        manager = cls.__managers__.get(h_node.file.id,None)
        if manager is not None:
            raise LookupError(
                "'{}' type manager already created for file '{}'".format(
                cls.__name__,h_node.file.filename
            )
        )
        table = cls.__managers__[h_node.file.id] = create_entry()
        weakref.finalize(table[0],cls._drop_manager,h_node.file.id)
        return table[0]

    @classmethod
    def get_manager(cls, h_node):
        """
        return manager responsible for the file containing h_node 
    
        Parameters
        ----------
        h_node (h5py.File, h5py.Group, h5py.Dataset):
            the h5py node to obtain the responsible manager for.
            
        Raises
        ------
        LookupError:
            if no manager has been created yet for h_node or its h_root_group 
        """
        try:
            return cls.__managers__[h_node.file.id][0]
        except KeyError:
            raise ReferenceError("no managers exist for file '{}'".format(h_node.file.filename))

    def __init__(self):
        if type.mro(self.__class__)[0] is BaseManager:
            raise TypeError("'BaseManager' class must be subclassed")

    def __enter__(self):
        raise NotImplementedError("'{}' type object must implement Python ContextManager protocol")

    def __exit__(self, exc_type, exc_value, exc_traceback, h_node=None):
        # remove this ReferenceManager object from the table of active ReferenceManager objects
        # and cleanly unlink from any h5py object instance and id references managed. Finalize
        # 'hickle_types_table' overlay if it was created by __init__ for hickle 4.0.X file
        self.__class__._drop_manager(h_node.file.id)

class ReferenceManager(BaseManager, dict):
    """
    Manages all object and type references created for basic
    and type special memoisation.

    To create a ReferenceManager call ReferenceManager.create_manager
    function. The value returned can be and shall be used within a
    with statement to ensure it is garbage collected before file is closed.
    For example:

        with ReferenceManager.create_manager(h_root_group) as memo:
            _dump(data,h_root_group,'data',memo,loader,**kwargs)

        with ReferenceManager.create_manager(h_root_group) as memo:
            _load(py_container,'data',h_root_group['data'],memo,loader)

        with ReferenceManager.create_manager(h_root_group,fix_lambda_obj_type) as memo:
            _load(py_container,'data',h_root_group['data'],memo,loader)

    NOTE: for creating appropriate loader object see LoaderManager
    """

    __slots__ = (
        '_py_obj_type_table', # hickle_types_table h5py.Group storing type information
        '_py_obj_type_link', # dictionary linking py_obj_type and representation in hickle_types_table
        '_base_type_link', # dictionary linking base_type string and representation in hickle_types_table
        '_overlay', # in memory hdf5 dummy file hosting dummy hickle_types_table for hickle 4.x files
        'pickle_loads' # reference to pickle.loads method
    )

    
    @staticmethod
    def get_root(h_node):
        """
        returns the h_root_group the passed h_node belongs to.
        """

        # try to resolve the 'type' attribute of the h_node
        entry_ref = h_node.attrs.get('type',None)
        if isinstance(entry_ref,h5.Reference):

            # return the grandparent of the referenced py_obj_type dataset as it
            # also the h_root_group of h_node
            try:
                entry = h_node.file.get(entry_ref,None)
            except ValueError: # pragma: no cover
                entry = None
            if entry is not None:
                return entry.parent.parent
        if h_node.parent == h_node.file:

            # h_node is either the h_root_group it self or the file node representing
            # the open hickle file. 
            return h_node if isinstance(h_node,h5.Group) else h_node.file

        # either h_node has not yet a 'type' assigned or contains pickle string
        # which has implicit b'pickle' type. try to resolve h_root_group from its
        # parent 'type' entry if any
        entry_ref = h_node.parent.attrs.get('type',None)
        if not isinstance(entry_ref,h5.Reference):
            if entry_ref is None:

                # parent has neither a 'type' assigned
                return h_node if isinstance(h_node,h5.Group) else h_node.file

            # 'type' seems to be a byte string or string fallback to h_node.file
            return h_node.file
        try:
            entry = h_node.file.get(entry_ref,None)
        except ValueError: # pragma: no cover
            entry = None
        if entry is None:

            # 'type' reference seems to be stale
            return h_node if isinstance(h_node,h5.Group) else h_node.file

        # return the grand parent of the referenced py_obj_type dataset as it
        # is also the h_root_group of h_node
        return  entry.parent.parent

    @staticmethod
    def _drop_overlay(h5file):
        """
        closes in memory overlay file providing dummy 'hickle_types_table' structure
        for hdf5 files which were created by hickle 4.x 
        """
        h5file.close()

    @classmethod
    def create_manager(cls,h_node, pickle_loads = pickle.loads):
        """
        creates a new ReferenceManager object for the h_root_group the h_node
        belongs to. 


        Parameters
        ----------
        h_node (h5py.Group, h5py.Dataset):
            the h5py node or its h_root_group to create a new ReferenceManager
            object for.

        pickle_loads (FunctionType,MethodType):
            method to be used to expand py_obj_type pickle strings.
            defaults to pickle.loads. Must be set to fix_lambda_obj_type 
            for hickle file created by hickle 4.x.

        Raises
        ------
        LookupError:
            if ReferenceManager has already been created for h_node or its h_root_group 
        """

        def create_manager():
            return ( 
                ReferenceManager(h_node,pickle_loads = pickle_loads),
                ReferenceManager.get_root(h_node)
            )
        return super().create_manager(h_node,create_manager)

    def __init__(self, h_root_group, *args,pickle_loads = pickle.loads, **kwargs):
        """
        constructs ReferenceManager object

        Parameters
        ----------
        h_root_group (h5py.Group):
            see ReferenceManager.create_manager

        args (tuple,list):
            passed to dict.__init__

        pickle_loads (FunctionType,MethodType):
            see ReferenceManager.create_manager

        kwargs (dict):
            passed to dict.__init__

        Raises
        ------
        ReferenceError:
            In case an error occurs while loading 'hickle_types_table' from an
            existing file opened for reading and writing
        """
        super().__init__(*args,**kwargs)
        self._py_obj_type_link = dict()
        self._base_type_link = dict()
        self._overlay = None
        self.pickle_loads = pickle_loads

        # get the 'hickle_types_table' member of h_root_group or create it anew
        # in case none found. In case hdf5 file is opened for reading only
        # create an in memory hdf5 file (managed by hdf5 'core' driver) providing
        # an empty dummy hickle_types_table. This is necessary to ensue that
        # ReferenceManager.resolve_type works properly on hickle 4.x files which
        # store type information directly in h5py.Group and h5py.Datasets attrs
        # structure.
        self._py_obj_type_table = h_root_group.get('hickle_types_table',None)
        if self._py_obj_type_table is None:
            if h_root_group.file.mode == 'r+':
                self._py_obj_type_table = h_root_group.create_group("hickle_types_table",track_order = True)
            else:
                h5_overlay = h5.File(
                    '{}.hover'.format(h_root_group.file.filename.rsplit('.',1)[0]),
                    mode='w', driver='core',backing_store=False
                )
                self._py_obj_type_table = h5_overlay.create_group("hickle_types_table",track_order = True)
                self._overlay = weakref.finalize(self,ReferenceManager._drop_overlay,h5_overlay)
            return

        # verify that '_py_obj_type_table' is a valid h5py.Group object
        if not isinstance(self._py_obj_type_table,h5.Group):
            raise ReferenceError("'hickle_types_table' invalid: Must be HDF5 Group entry")

        # if h_root_group.file was opened for writing restore '_py_obj_type_link' and
        # '_base_type_link' table entries from '_py_obj_type_table' to ensure when
        # h5py.Group and h5py.Dataset are added anew to h_root_group tree structure
        # their 'type' attribute is set to the correct py_obj_type reference by the
        # ReferenceManager.store_type method. Each of '_py_obj_type_link' and
        # '_base_type_link' tables can be used to properly restore  the 'py_obj_type'
        # and 'base_type' when loading the file as well as assigning to the 'type'
        # attribute the appropriate 'py_obj_type' dataset reference from the
        # '_py_obj_type_table' when dumping data to the file.
        if h_root_group.file.mode != 'r+':
            return
        for _, entry in self._py_obj_type_table.items():
            if entry.shape is None and entry.dtype == 'S1':
                base_type = entry.name.rsplit('/',1)[-1].encode('ascii')
                self._base_type_link[base_type] = entry
                self._base_type_link[entry.id] = base_type
                continue
            base_type_ref = entry.attrs.get('base_type',None)
            if not isinstance(base_type_ref,h5.Reference):
                raise ReferenceError(
                    "inconsistent 'hickle_types_table' entries for py_obj_type '{}': "
                    "no base_type".format(py_obj_type)
                )
            try:
                base_type_entry = entry.file.get(base_type_ref,None)
            except ValueError: # pragma: no cover
                base_type_entry = None
            if base_type_entry is None:
                raise ReferenceError(
                    "inconsistent 'hickle_types_table' entries for py_obj_type '{}': "
                    "stale base_type".format(py_obj_type)
                )
            base_type = self._base_type_link.get(base_type_entry.id,None)
            if base_type is None:
                base_type = base_type_entry.name.rsplit('/',1)[-1].encode('ascii')
            try:
                py_obj_type = pickle.loads(entry[()])
            except (ImportError,AttributeError):
                py_obj_type = AttemptRecoverCustom
                entry_link = py_obj_type,'!recover!',base_type
            else:
                entry_link = py_obj_type,base_type
                self._py_obj_type_link[id(py_obj_type)] = entry
            self._py_obj_type_link[entry.id] = entry_link

    def store_type(self, h_node, py_obj_type, base_type = None, attr_name = 'type', **kwargs):
        """
        assigns a 'py_obj_type' entry reference to the attribute specified by attr_name
        of h_node and creates if not present the appropriate 'hickle_types_table'
        entries for py_obj_type and base_type.

        Note
        ----
            Storing and restoring the content of nodes containing pickle byte strings
            is fully managed by pickle.dumps and pickle.loads functions including
            selection of appropriate py_obj_type. Therefore no explicit entry for 
            object and b'pickle' py_obj_type and base_type pairs indicating pickled
            content of pickled dataset are created.

        Parameters
        ----------
        h_node (h5py.Group, h5py.Dataset):
            node the 'type' attribute a 'hickle_types_table' entry corresponding to 
            the provided py_obj_type, base_type entry pair shall be assigned to.

        py_obj_type (any type or class):
            the type or class of the object instance represented by h_node

        base_type (bytes):
            the base-type bytes string of the loader used to create the h_node and
            restore an object instance form on load. If None no 'hickle_types_table'
            will be crated for py_obj_type if not already present and a LookupError
            exception is raised instead.

        attr_name (str):
            the name of the attribute the type reference shall be stored to. Defaults
            to 'type'

        kwargs (dict):
            keyword arguments to be passed to h5py.Group.create_dataset function
            when creating the entries for py_obj_type and base_type anew

        Raises
        ------
        ValueError:
            if base_type is not a valid bytes string

        LookupError:
            if base_type is None and no 'hickle_types_table' entry exists for
            py_obj_type yet

        """

        # return immediately if py_obj_type is object as h_node contains pickled byte
        # string of the actual object dumped
        if py_obj_type is object:
            return

        # if no entry within the 'hickle_types_table' exists yet
        # for py_obj_type create the corresponding pickle string dataset
        # and store appropriate entries in the '_py_obj_type_link' table for
        # further use by ReferenceManager.store_type and ReferenceManager.resolve_type
        # methods
        py_obj_type_id = id(py_obj_type)
        entry = self._py_obj_type_link.get(py_obj_type_id,None)
        if entry is None:
            if base_type is None:
                raise LookupError(
                    "no entry found for py_obj_type '{}'".format(py_obj_type.__name__)
                ) 
            if not isinstance(base_type,(str,bytes)) or not base_type:
                raise ValueError("base_type must be non empty bytes string")
            type_entry = memoryview(pickle.dumps(py_obj_type))
            type_entry = np.array(type_entry,copy = False)
            type_entry.dtype = 'S1'
            entry = self._py_obj_type_table.create_dataset(
                str(len(self._py_obj_type_table)),
                data=type_entry,
                shape=(1,type_entry.size),
                **kwargs
            )
            # assign a reference to base_type entry within 'hickle_types_table' to
            # the 'base_type' attribute of the newly created py_obj_type entry.
            # if 'hickle_types_table' does not yet contain empty dataset entry for
            # base_type create it and store appropriate entries in the '_base_type_link' table
            # for further use by ReferenceManager.store_type and ReferenceManager,resolve_type
            # methods
            base_entry = self._base_type_link.get(base_type,None)
            if base_entry is None:
                base_entry = self._base_type_link[base_type] = self._py_obj_type_table.create_dataset(
                    base_type.decode('ascii'),
                    shape=None,dtype = 'S1',
                    **no_compression(kwargs)
                )
                self._base_type_link[base_entry.id] = base_type
            entry.attrs['base_type'] = base_entry.ref
            self._py_obj_type_link[py_obj_type_id] = entry
            self._py_obj_type_link[entry.id] = (py_obj_type,base_type)
        h_node.attrs[attr_name] = entry.ref

    def resolve_type(self,h_node,attr_name = 'type',base_type_type = 1):
        """
        resolves the py_obj_type and base_type pair referenced to by the 'type' attribute and
        if present the 'base_type' attribute.

        Note: If the 'base_type' attribute is present it is assumed that the dataset was
            created by hickle 4.x version. Consequently it is assumed that the 'type' attribute
            contains a pickle bytes string to load the py_obj_type from instead of a reference
            to a 'hickle_types_table' entry representing the py_obj_type base_type pair of h_node.

        Note: If 'type' attribute is not present h_node represents either a h5py.Reference to
            the actual node of the object to be restored or contains a pickle bytes string.
            In either case the corresponding implicit py_obj_type base_type pair 
            (NodeReference, b'!node-reference!') or (object,b'pickle') respective is assumed and
            returned.

        Note: If restoring 'py_object_type' from pickle string stored in type attribute or 
            'hickle_types_table' fails than implicit py_obj_type base_type pair 
            (AttemptRecoverCustom,'!recover!') is returned instead of the actual 'py_obj_type' 
            base_type pair is returned. The latter can be retrieved by setting 'base_type_type'
            to 2 in this case

        Parameters
        ----------
        h_node (h5py.Group,h5py.Dataset):
            the node to resolve py_obj_type and base_type for using reference stored in
            attribute specified by attr_name

        attr_name (str):
            the name of the attribute the type reference shall be restored from. Defaults
            to and must be 'type' in case not a h5py.Reference.

        base_type_type (int):
            1 (default) base_type used to select loader
           -1 original base_type corresponding to not understood py_obj_type of
              recovered h5py.Group or h5py.Dataset

        Returns
        -------
        tuple containing (py_obj_type,base_type,is_container)

        py_obj_type:
            the python type of the restored object

        base_type:
            the base_type string indicating the loader to be used for properly
            restoring the py_obj_type instance or the base_type string 

        is_container:
            boolean flag indicating whether h_node represents a h5py.Group or
            h5py.Reference both of which have to be handled by corresponding
            PyContainer type loaders or a h5py.Dataset for which the appropriate
            load_fn is to be called.
        """

        # load the type attribute indicated by attr_name. If not present check if h_node
        # is h5py.Reference dataset or a dataset containing a pickle bytes
        # string. In either case assume (NodeReference,b'!node-reference!') or (object,b'pickle')
        # respective as (py_obj_type, base_type) pair and set is_container flag to True for
        # h5py.Reference and False otherwise.
        # 
        # NOTE: hickle 4.x legacy file does not store 'type' attribute for h5py.Group nodes with 
        #       b'dict_item' base_type. As h5py.Groups do not have a dtype attribute the check
        #       whether h_node.dtype equals h5py.ref_dtype will raise AttributeError.
        #       If h_node represents a b'dict_item' than self.pickle_loads will point to
        #       fix_lambda_obj_type below which will properly handle None value of type_ref
        #       in any other case file is not a hickle 4.x legacy file and thus has to be
        #       considered broken
        type_ref = h_node.attrs.get(attr_name,None)
        if not isinstance(type_ref,h5.Reference):
            if type_ref is None:
                try:
                    metadata = h_node.dtype.metadata
                except (AttributeError,):
                    pass
                else:
                    if metadata is not None and issubclass(metadata.get('ref',object),h5.Reference):
                        return NodeReference,b'!node-reference!',True
                    return object,b'pickle',False

            # check if 'type' attribute of h_node contains a reference to a 'hickle_types_table'
            # entry. If not use pickle to restore py_object_type from the 'type' attribute value
            # directly if possible
            try:

                # set is_container_flag to True if h_node is h5py.Group type object and false
                # otherwise
                return self.pickle_loads(type_ref), h_node.attrs.get('base_type', b'pickle'), isinstance(h_node, h5.Group)
            except (ModuleNotFoundError,AttributeError):
                # module missing or py_object_type not provided by module
                return AttemptRecoverCustom,( h_node.attrs.get('base_type',b'pickle') if base_type_type == 2 else b'!recover!' ),isinstance(h_node,h5.Group)
            except (TypeError, pickle.UnpicklingError, EOFError):
                raise ReferenceError(
                    "node '{}': '{}' attribute ('{}')invalid: not a pickle byte string".format(
                        h_node.name,attr_name,type_ref
                    )
                )
        try:
            entry = self._py_obj_type_table[type_ref]
        except (ValueError, KeyError):
            raise ReferenceError(
                "node '{}': '{}' attribute invalid: stale reference".format(
                    h_node.name,attr_name
                )
            )

        # load (py_obj_type,base_type) pair from _py_obj_type_link for 'hickle_types_table' entry
        # referenced by 'type' entry. Create appropriate _py_obj_type_link and _base_type_link
        # entries if if not present for (py_obj_type,base_type) pair for further use by
        # ReferenceManager.store_type and ReferenceManager.resolve_type methods.
        type_info = self._py_obj_type_link.get(entry.id, None)
        if type_info is None:
            base_type_ref = entry.attrs.get('base_type', None)
            if base_type_ref is None:
                base_type = b'pickle'
            else:
                try:
                    base_type_entry = self._py_obj_type_table[base_type_ref]
                except ( ValueError,KeyError ):
                    # TODO should be recovered here instead?
                    raise ReferenceError(
                        "stale base_type reference encountered for '{}' type table entry".format(
                            entry.name
                        )
                    )
                base_type = self._base_type_link.get(base_type_entry.id,None)
                if base_type is None:

                    # get the relative table entry name form full path name of entry node
                    base_type = base_type_entry.name.rsplit('/',1)[-1].encode('ASCII')
                    self._base_type_link[base_type] = base_type_entry
                    self._base_type_link[base_type_entry.id] = base_type
            try:
                py_obj_type = self.pickle_loads(entry[()])
            except (ModuleNotFoundError,AttributeError):
                py_obj_type = AttemptRecoverCustom
                entry_link = (py_obj_type,b'!recover!',base_type)
            else:
                entry_link = (py_obj_type,base_type)
                self._py_obj_type_link[id(py_obj_type)] = entry
            type_info = self._py_obj_type_link[entry.id] = entry_link
        # return (py_obj_type,base_type). set is_container flag to true if 
        # h_node is h5py.Group object and false otherwise
        return (type_info[0],type_info[base_type_type],isinstance(h_node,h5.Group))

    def __enter__(self):
        if not isinstance(self._py_obj_type_table, h5.Group) or not self._py_obj_type_table:
            raise RuntimeError(
                "Stale ReferenceManager, call ReferenceManager.create_manager to create a new one"
            )
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if not isinstance(self._py_obj_type_table, h5.Group) or not self._py_obj_type_table:
            return

        # remove this ReferenceManager object from the table of active ReferenceManager objects
        # and cleanly unlink from any h5py object instance and id references managed. Finalize
        # 'hickle_types_table' overlay if it was created by __init__ for hickle 4.x file
        super().__exit__(exc_type, exc_value, exc_traceback, self._py_obj_type_table)
        self._py_obj_type_table = None
        self._py_obj_type_link = None
        self._base_type_link = None
        self.pickle_loads = None
        if self._overlay is not None:
            self._overlay()
            self._overlay = None

#####################
# loading optional  #
#####################

_managed_by_hickle = {'hickle', ''}

_custom_loader_enabled_builtins = {'__main__':('','')}

class LoaderManager(BaseManager):
    """
    Handles the file specific lookup of loader to be used to dump or load
    a python object of a specific type 

    To create a LoaderManager call LoaderManager.create_manager
    function. The value returned can be and shall be used within a
    with statement for example as follows:

        with LoaderManager.create_manager(h_root_group) as loader:
            _dump(data,h_root_group,'data',memo,loader,**kwargs)

        with LoaderManager.create_manager(h_root_group,False,{'custom':true}) as loader:
            _load(py_container,'data',h_root_group['data'],memo,loader)

        with LoaderManager.create_manager(h_root_group,True) as memo:
            _load(py_container,'data',h_root_group['data'],memo,loader)

    NOTE: for creating appropriate memo object see ReferenceManager

    """

    # Define dict of all acceptable types dependent upon loader option
    __py_types__ = { 
        None: {},
        'hickle-4.x': {},
        **{ option:{} for option in optional_loaders }
    }

    # Define dict of all acceptable load function dependent upon loader option
    __hkl_functions__ = {
        None: {},
        'hickle-4.x': {},
        **{ option:{} for option in optional_loaders }
    }

    # Define dict of all acceptable hickle container types dependent upon loader option
    __hkl_container__ = {
        None: {},
        'hickle-4.x': {},
        **{ option:{} for option in optional_loaders }
    }

    # Empty list (hashtable) of loaded loader names
    __loaded_loaders__ = set()


    @classmethod
    def register_class(
        cls, myclass_type, hkl_str, dump_function=None, load_function=None,
        container_class=None, memoise = True, option=None
    ):
        """
        Register a new class to be recognized and dumped or restored by hickle.
    
        Parameters
        ----------
        myclass_type (type.class):
            the class to register dump_fcn, load_fcn, PyContainer for

        hkl_str (str):
            String to write to HDF5 file identifying class and loader suitable
            for restoring the py_object described by the data stored in hickle
            file.

            NOTE: dict_item, pickle, !node-refrence!, !recover! and any other
                string enclosed within a pair of !! can not be modified if once
                registered. Strings quoted by !! must be added as global loader
                with option = None.
                
        dump_function (callable):
            callable to write data to HDF5

        load_function (callable):
            function to load data from HDF5

        container_class (PyContainer):
            PyContainer type proxy class to load data from HDF5

        memoise (bool): 
            True: references to the object instances of class shall be remembered
                during dump and load for properly resolving multiple references
                to the same object instance.

            False: every occurrence of an instance of the object has to be dumped
                and restored on load disregarding instances already present.

        option (str, None):
            String identifying set of loaders which shall only be used when
            specific feature or category is requested on top of global loaders.
            If None than loader is globally to be used if there is no other loader
            registered for myclass_type. 

            NOTE: only strings listed in 'optional_loaders' exported by 
                  hickle.loaders.__init__ and 'hickle-4.x' are accepted. 
                
        Raises
        ------
        TypeError:
            loader for myclass_type may only be registered by hickle core modules not
            loaded from hickle/loaders/ directory, <packagepath>/hickle_loaders/,
            <modulepath>/hickle_loaders/ or <__main__path>/hickle_loaders/ directory
            by explicitly calling LoaderManager.register_class method.

        ValueError:
            if optional loader modules tries to shadow 'dict_item', 'pickle' and any
            loader marked as essential to proper function of hickle.dump and hickle.load 
            by ! prefix and postfix ('!node-reference!', '!recover!').

        LookupError:
            if optional loader denoted by option is unknown. Any new option must be
            listed in 'optional_loaders' exported by 'hickle.loaders.__init__.py'
            file to be recognized as valid option
        """
    
        if (
            myclass_type is object or
            isinstance(
                myclass_type,
                (types.FunctionType, types.BuiltinFunctionType, types.MethodType, types.BuiltinMethodType)
            ) or
            issubclass(myclass_type,(type,_DictItem))
        ):
            # object, all functions, methods, class objects and the special _DictItem class
            # type objects are to be handled by hickle core only. 
            dump_module = getattr(dump_function, '__module__', '').split('.', 2)
            load_module = getattr(load_function, '__module__', '').split('.', 2)
            container_module = getattr(container_class, '__module__', '').split('.', 2)
            if {dump_module[0], load_module[0], container_module[0]} - _managed_by_hickle:
                raise TypeError(
                    "loader for '{}' type managed by hickle only".format(
                        myclass_type.__name__
                    )
                )
            if "loaders" in {*dump_module[1:2], *load_module[1:2], *container_module[1:2]}:
                raise TypeError(
                    "loader for '{}' type managed by hickle core only".format(
                        myclass_type.__name__
                    )
                )
        if ( 
            ( cls.__hkl_functions__[None].get(hkl_str) or cls.__hkl_container__[None].get(hkl_str) ) and
            ( hkl_str[:1] == hkl_str[-1:] == b'!' or hkl_str in disallow_in_option )
        ):
            raise ValueError(
                "'{}' base_type may not be shadowed by loader".format(hkl_str)
            )
        # add loader
        try:
            if dump_function is not None:
                cls.__py_types__[option][myclass_type] = ( dump_function, hkl_str,memoise)
            if load_function is not None:
                cls.__hkl_functions__[option][hkl_str] = load_function
                cls.__hkl_functions__[option][hkl_str.decode('ascii')] = load_function
            if container_class is not None:
                cls.__hkl_container__[option][hkl_str] = container_class
                cls.__hkl_container__[option][hkl_str.decode('ascii')] = container_class
        except KeyError:
            raise LookupError("Invalid option '{}' encountered".format(option))
        
            


    @classmethod
    def register_class_exclude(cls, hkl_str_to_ignore, option = None):
        """
        Tell loading function to ignore any HDF5 dataset with attribute 'type=XYZ'
    
        Parameters
        ----------
        hkl_str_to_ignore (str): attribute type=string to ignore and exclude
            from loading.

        option (str, None):
            String identifying set of optional loaders from which class shall
            be excluded
                
        Raises
        ------
        ValueError:
            class is managed by hickle core machinery and thus may not be ignored

        LookupError:
            option loader shall belong to is unknown. Any new option must be listed
            in 'optional_loaders' exported by 'hickle.loaders.__init__.py' file to be
            recognized as valid option
        """
    
        if hkl_str_to_ignore[0] == hkl_str_to_ignore[-1] == b'!' or hkl_str_to_ignore in disallowed_to_ignore:
            raise ValueError(
                "excluding '{}' base_type managed by hickle core not possible".format(
                    hkl_str_to_ignore
                )
            )
        try:
            cls.__hkl_functions__[option][hkl_str_to_ignore] = load_nothing
            cls.__hkl_container__[option][hkl_str_to_ignore] = NoContainer
            cls.__hkl_functions__[option][hkl_str_to_ignore.decode('ascii')] = load_nothing
            cls.__hkl_container__[option][hkl_str_to_ignore.decode('ascii')] = NoContainer
        except KeyError:
            raise LookupError("'{}' option unknown".format(option))

    __slots__ = ( 'types_dict', 'hkl_types_dict', 'hkl_container_dict', '_mro', '_file')


    _option_formatter = '{}{{}}'.format(attribute_prefix)
    _option_parser = re.compile(r'^{}(.*)$'.format(attribute_prefix),re.I)

    def __init__(self, h_root_group, legacy = False, options = None):
        """
        constructs LoaderManager object

        Parameters
        ----------
        h_root_group (h5py.Group):
            see LoaderManager.create_manager

        legacy (bool):
            If true the file h_node belongs to is in legacy hickle 4.x format.
            Ensure lambda py_obj_type strings are loaded properly and
            'hickle-4.x' type loaders are included within types_dict, 
            'hkl_types_dict' and 'hkl_container_dict'

        options (dict):
            optional loaders to be loaded. Each key names one loader and
            its value indicates whether to be used (True) or excluded (False) 

        Raises
        ------
        LookupError:
            option loader unknown
        """

        # initialize lookup dictionaries with set of common loaders
        self.types_dict = collections.ChainMap(self.__class__.__py_types__[None])
        self.hkl_types_dict = collections.ChainMap(self.__class__.__hkl_functions__[None])
        self.hkl_container_dict = collections.ChainMap(self.__class__.__hkl_container__[None])

        # Select source of optional loader flags. If option is None try to read options
        # from h_root_group.attrs structure. Otherwise use content of options dict store
        # each entry to be used within h_root_group.attrs structure or update entry there
        if options is None:
            option_items = ( 
                match.group(1).lower() 
                for match,on in ( 
                    ( LoaderManager._option_parser.match(name), value )
                    for name, value in h_root_group.attrs.items() 
                ) 
                if match and on
            )
        else:
            def set_option_items():
                for option_key,on in options.items():
                    if not on:
                        continue
                    h_root_group.attrs[LoaderManager._option_formatter.format(option_key.upper())] = on 
                    yield option_key
            option_items = set_option_items()
        # try to include loader set indicated by option_name
        try:
            for option_name in option_items:
                self.types_dict.maps.insert(0,self.__class__.__py_types__[option_name])
                self.hkl_types_dict.maps.insert(0,self.__class__.__hkl_functions__[option_name])
                self.hkl_container_dict.maps.insert(0,self.__class__.__hkl_container__[option_name])
        except KeyError:
            raise LookupError("Option '{}' invalid".format(option_name))
            
        # add loaders required to properly load legacy files created by hickle 4.x and
        # ensure that non class types are properly reported by load_loader
        if legacy:
            self._mro = type_legacy_mro
            self.types_dict.maps.insert(0,self.__class__.__py_types__['hickle-4.x'])
            self.hkl_types_dict.maps.insert(0,self.__class__.__hkl_functions__['hickle-4.x'])
            self.hkl_container_dict.maps.insert(0,self.__class__.__hkl_container__['hickle-4.x'])
        else:
            self._mro = type.mro
        self._file = h_root_group.file
        
    def load_loader(self, py_obj_type):
        """
        Checks if given `py_obj` requires an additional loader to be handled
        properly and loads it if so.
    
        Parameters
        ----------
        py_obj:
            the Python object to find an appropriate loader for
    
        Returns
        -------
        tuple containing (py_obj, (create_dataset, base_type, memoise))

        py_obj:
             the Python object the loader was requested for
    
        (create_dataset,base_type,memoise):
             tuple providing create_dataset function, name of base_type used
             to represent py_obj and the boolean memoise flag indicating
             whether loaded object shall be remembered for restoring further
             references to it or must be loaded every time encountered.
    
        Raises
        ------
        RuntimeError:
            in case py object is defined by hickle core machinery.
        """
    
        types_dict = self.types_dict
        loaded_loaders = self.__class__.__loaded_loaders__
        # loop over the entire mro_list of py_obj_type
        for mro_item in self._mro(py_obj_type):

            # Check if mro_item is already listed in types_dict and return if found 
            loader_item = types_dict.get(mro_item,None)
            if loader_item is not None:
                return py_obj_type,loader_item
    
            # Obtain the package name of mro_item
            package_list = mro_item.__module__.split('.',2)
    
            package_file = None 
            if package_list[0] == 'hickle':
                if package_list[1] != 'loaders':
                    print(mro_item,package_list)
                    raise RuntimeError(
                        "objects defined by hickle core must be registered"
                        " before first dump or load"
                    )
                if (
                    len(package_list) < 3 or
                    not package_list[2].startswith("load_") or
                    '.' in package_list[2][5:]
                ):
                    warnings.warn(
                        "ignoring '{!r}' dummy type not defined by loader module".format(py_obj_type),
                        RuntimeWarning
                    )
                    continue

                # dummy objects are not dumpable ensure that future lookups return that result
                loader_item = types_dict.get(mro_item,None)
                if loader_item is None:
                    loader_item = types_dict[mro_item] = ( not_dumpable, b'NotHicklable',False )

                # ensure module of mro_item is loaded as loader as it will contain
                # loader which knows how to handle group or dataset with dummy as 
                # py_obj_type 
                loader_name = mro_item.__module__
                if loader_name in loaded_loaders:

                    # loader already loaded as triggered by dummy abort search and return
                    # what found so far as fallback to further bases does not make sense
                    return py_obj_type,loader_item
            else:
                loader_name,package_file = _custom_loader_enabled_builtins.get(package_list[0],(None,''))
                if loader_name is None:
                    # construct the name of the associated loader
                    loader_name = 'hickle.loaders.load_{:s}'.format(package_list[0])
                elif not loader_name:
                    # try to resolve module name for __main__ script and other generic modules
                    package_module = sys.modules.get(package_list[0],None)
                    if package_module is None:
                        warnings.warn(
                            "package/module '{}' defining '{}' type dropped".format(
                                package_list[0],mro_item.__name__
                            ),
                            PackageImportDropped
                        )
                        continue
                    package_file = getattr(package_module,'__file__',None)
                    if package_file is None:
                        package_loader = getattr(package_module,'__loader__',None)
                        if package_loader is None: # pragma: no cover
                            # just to secure against "very smart" tinkering
                            # with python import machinery, no serious testable use-case known and expected
                            continue
                        package_spec = spec_from_loader(package_list[0],package_loader)
                        if not getattr(package_spec,'has_location',False):
                            continue
                        package_file = package_spec.origin
                    if not os.path.isabs(package_file): # pragma: no cover
                        # not sure if this case wouldn't just be result of "very smart" tinkering
                        # with python import machinery, no serious testable use-case known yet
                        package_spec = find_spec(os.path.basename(package_file.rsplit('.')[0]))
                        if not getattr(package_spec,'has_location',False): # pargma: no cover
                            # not sure if this case wouldn't just be result of "very smart" tinkering
                            # with python import machinery, no serious testable use-case known yet
                            continue
                        package_file = package_spec.origin
                    package_list[0],allow_custom_loader = os.path.basename(package_file).rsplit('.')[0],package_list[0]
                    loader_name = 'hickle.loaders.load_{:s}'.format(package_list[0])
                    _custom_loader_enabled_builtins[allow_custom_loader] = loader_name, package_file
    
                # Check if this module is already loaded
                if loader_name in loaded_loaders:
                    # loader is loaded but does not define loader for mro_item
                    # check next base class
                    continue
    
            # check if loader module has already been loaded. If use that instead
            # of importing it anew
            loader = sys.modules.get(loader_name,None)
            if loader is None:
                # Try to load a loader with this name
                loader_spec = find_spec(loader_name)
                if loader_spec is None:
                    assert isinstance(package_file,str), "package_file name for _custom_loader_enabled_builtins must be string"
                    if not package_file:
                        package_spec = getattr(sys.modules.get(package_list[0],None),'__spec__',None)
                        if package_spec is None:
                            package_spec = find_spec(package_list[0])
                        if not getattr(package_spec,'has_location',False):
                            # can't resolve package or base module hosting mro_item
                            continue
                    package_path = os.path.dirname(package_file)
                    package_loader_path = os.path.join(
                        package_path, "hickle_loaders", "load_{:s}.py".format(package_list[0])
                    )
                    try:
                        fid = open(package_loader_path,'rb')
                    except FileNotFoundError:
                        try:
                            package_loader_path += 'c'
                            fid = open(package_loader_path,'rb')
                        except FileNotFoundError:
                            # no file for loader module found
                            continue
                        else:
                            fid.close()
                    else:
                        fid.close()
                    loader_spec = spec_from_file_location(loader_name,package_loader_path)
                # import the the loader module described by module_spec
                # any import errors and exceptions result at this stage from
                # errors inside module and not cause loader module does not
                # exist
                loader = module_from_spec(loader_spec)
                loader_spec.loader.exec_module(loader)
                sys.modules[loader_name] = loader
    
            # load all loaders defined by loader module
            for next_loader in loader.class_register:
                self.register_class(*next_loader)
            for drop_loader in ( loader if isinstance(loader,(list,tuple)) else (loader,None) for loader in loader.exclude_register) :
                self.register_class_exclude(*drop_loader)
            loaded_loaders.add(loader_name)
    
            # check if loader module defines a loader for base_class mro_item
            loader_item = types_dict.get(mro_item,None)
            if loader_item is None:
                # the new loader does not define loader for mro_item
                # check next base class
                continue

            # return loader for base_class mro_item
            return py_obj_type,loader_item
    
        # no appropriate loader found return fallback to pickle
        return py_obj_type,(create_pickled_dataset,b'pickle',True)

    @classmethod
    def create_manager(cls, h_node, legacy = False, options = None):
        """
        creates an new LoaderManager object for the h_root_group the h_node
        belongs to. 


        Parameters
        ----------
        h_node (h5py.Group, h5py.Dataset):
            the h5py node or its h_root_group to create a new LoaderManager
            object for.

        legacy (bool):
            if true file h_node belongs to is in legacy hickle 4.x format
            ensure lambda py_obj_type strings are loaded properly and
            'hickle-4.x' type loaders are included within types_dict, 
            'hkl_types_dict' and 'hkl_container_dict'

        options (dict):
            optional loaders to be loaded. Each key names one loader and
            its value indicates whether to be used (True) or excluded (False) 

        Raises
        ------
        LookupError:
            if ReferenceManager has already been created for h_node or its h_root_group 
        """

        def create_manager():
            return (LoaderManager(h_node,legacy,options),)
        return super().create_manager(h_node,create_manager)

    def __enter__(self):
        if not isinstance(self._file,h5.File) or not self._file:
            raise RuntimeError(
                "Stale LoaderManager, call LoaderManager.create_manager to create a new one"
            )
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if not isinstance(self._file,h5.File) or not self._file:
            return
        super().__exit__(exc_type, exc_value, exc_traceback, self._file)
        self._file = None
        self._mro = None
        self.types_dict = None
        self.hkl_types_dict = None
        self.hkl_container_dict = None
        
def type_legacy_mro(cls):
    """
    drop in replacement of type.mro for loading legacy hickle 4.x files which were
    created without generalized PyContainer objects available. Consequently some
    h5py.Datasets and h5py.Group objects expose function objects as their py_obj_type
    type.mro expects classes only.

    Parameters
    ----------
    cls (type):
        the py_obj_type/class of the object to load or dump

    Returns
    -------
        mro list for cls as returned by type.mro  or in case cls is a function or method
        a single element tuple is returned
    """
    if isinstance(
        cls,
        (types.FunctionType,types.BuiltinFunctionType,types.MethodType,types.BuiltinMethodType)
    ):
        return (cls,)
    return type.mro(cls) 

# %% BUILTIN LOADERS (not maskable)

# list of below hkl_types which may not be ignored
# NOTE: types which are enclosed in !! pair are disallowed in any case
disallowed_to_ignore = {b'dict_item', b'pickle' }

# list of below hkl_types which may not be redefined by optional loader
# NOTE: types which are enclosed in !! pair are disallowed in any case
disallow_in_option = {b'pickle'}

class NoContainer(PyContainer): # pragma: no cover
    """
    load nothing container
    """

    def convert(self):
        pass


class _DictItemContainer(PyContainer):
    """
    PyContainer reducing hickle version 4.x dict_item type h5py.Group to 
    its content for inclusion within dict h5py.Group
    """

    def convert(self):
        return self._content[0]

LoaderManager.register_class(
    _DictItem, b'dict_item', dump_nothing, load_nothing, _DictItemContainer, False, 'hickle-4.x'
)

        
class ExpandReferenceContainer(PyContainer):
    """
    PyContainer for properly restoring additional references to an object
    instance shared multiple times within the dumped object structure
    """

    def filter(self,h_parent):
        """
        resolves the h5py.Reference link and yields the the node
        it refers to as sub item of h_parent so that it can be
        properly loaded by recursively calling hickle._load method
        independent whether it can be directly loaded from the memo
        dictionary or has to be restored from file. 
        """

        try:
            referred_node = h_parent.file.get(h_parent[()],None)
        except ( ValueError, KeyError ): # pragma no cover
            referred_node = None
        if referred_node is None:
            raise ReferenceError("node '{}' stale node reference".format(h_parent.name))
        yield referred_node.name.rsplit('/',1)[-1], referred_node

    def convert(self):
        """
        returns the object the reference was pointing to
        """

        return self._content[0]

# objects created by resolving h5py.Reference datasets are already stored inside
# memo dictionary so no need to memoise them.
LoaderManager.register_class(
    NodeReference, b'!node-reference!', dump_nothing, load_nothing, ExpandReferenceContainer, False
)


def create_pickled_dataset(py_obj, h_group, name, reason = None, **kwargs):
    """
    Create pickle string as object can not be mapped to any other h5py
    structure.

    Parameters
    ----------
    py_obj:
        python object to dump; default if item is not matched.

    h_group (h5.File.group):
        group to dump data into.

    name (str):
        the name of the resulting dataset

    reason (str,None):
        reason why py_object has to be pickled eg. string
        provided by NotHicklable exception

    Warnings
    -------
    SerializedWarning:
        issued before pickle string is created

    """

    # for what ever reason py_obj could not be successfully reduced
    # ask pickle for help and report to user.
    reason_str = " (Reason: %s)" % (reason) if reason is not None else ""
    warnings.warn(
        "{!r} type not understood, data is serialized:{:s}".format(
            py_obj.__class__.__name__, reason_str
        ),
        SerializedWarning
    )

    # store object as pickle string
    pickled_obj = pickle.dumps(py_obj)
    d = h_group.create_dataset(name, data = memoryview(pickled_obj), **kwargs)
    return d,() 

def load_pickled_data(h_node, base_type, py_obj_type):
    """
    loade pickle string and return resulting py_obj
    """
    try:
        return pickle.loads(h_node[()])
    except (ImportError,AttributeError):
        return RecoveredDataset(h_node[()],dtype = h_node.dtype,attrs = dict(h_node.attrs))

        
# no dump method is registered for object as this is the default for
# any unknown object and for classes, functions and methods
LoaderManager.register_class(object,b'pickle',None,load_pickled_data)

def recover_custom_dataset(h_node,base_type,py_obj_type):
    """
    drop in load_fcn for any base_type no appropriate loader could be found
    """
    manager = ReferenceManager.get_manager(h_node)
    _,base_type,_ = manager.resolve_type(h_node,base_type_type = -1)
    warnings.warn(
        "loader '{}' missing for '{}' type object. Data recovered ({})".format(
            base_type, 
            py_obj_type.__name__ if not isinstance(py_obj_type, AttemptRecoverCustom) else None,
            h_node.name.rsplit('/')[-1]
        ),
        DataRecoveredWarning
    )
    attrs = dict(h_node.attrs)
    attrs['base_type'] = base_type
    return RecoveredDataset(h_node[()],dtype=h_node.dtype,attrs=attrs)

class RecoverGroupContainer(PyContainer):
    """
    drop in PyContainer for any base_type not appropriate loader could be found
    """
    def __init__(self,h5_attrs, base_type, object_type):
        super().__init__(h5_attrs, base_type, object_type,_content = {})
    
    def filter(self,h_parent):
        """
        switch base_type to the one loader is missing for
        """

        warnings.warn(
            "loader '{}' missing for '{}' type object. Data recovered ({})".format(
                self.base_type,
                self.object_type.__name__ if not isinstance(self.object_type,AttemptRecoverCustom) else None,
                h_parent.name.rsplit('/')[-1]
            ),
            DataRecoveredWarning
        )
        manager = ReferenceManager.get_manager(h_parent)
        _,self.base_type,_ = manager.resolve_type(h_parent,base_type_type = -1)
        yield from h_parent.items()

    def append(self,name,item,h5_attrs):
        if isinstance(item,AttemptRecoverCustom):
            self._content[name] = item
        else:
            self._content[name] = (item,{ key:value for key,value in h5_attrs.items() if key not in {'type'}})

    def convert(self):
        attrs = {key:value for key,value in self._h5_attrs.items() if key not in {'type'}}
        attrs['base_type'] = self.base_type
        return RecoveredGroup(self._content,attrs=attrs)


LoaderManager.register_class(AttemptRecoverCustom,b'!recover!',None,recover_custom_dataset,RecoverGroupContainer,True)
    

def _moc_numpy_array_object_lambda(x):
    """
    drop in replacement for lambda object types which seem not
    any more be accepted by pickle for Python 3.8 and onward.
    see fix_lambda_obj_type function below

    Parameters
    ----------
    x (list):
        itemlist from which to return first element

    Returns
    -------
    first element of provided list
    """
    return x[0]

LoaderManager.register_class(
    _moc_numpy_array_object_lambda, b'!moc_lambda!', dump_nothing, load_nothing, None, True, 'hickle-4.x'
)

def fix_lambda_obj_type(bytes_object, *, fix_imports=True, encoding="ASCII", errors="strict"):
    """
    drop in replacement for pickle.loads method when loading files created by hickle 4.x 
    It captures any TypeError thrown by pickle.loads when encountering a pickle string 
    representing a lambda function used as py_obj_type for a h5py.Dataset or h5py.Group.
    While in Python <3.8 pickle loads creates the lambda Python >= 3.8 throws an 
    error when encountering such a pickle string. This is captured and
    _moc_numpy_array_object_lambda returned instead. Further some h5py.Group and h5py.Datasets
    do not provide any py_obj_type for them object is returned assuming that proper loader has
    been identified by other objects already
    """
    if bytes_object is None:
        return object
    try:
        return pickle.loads(bytes_object, fix_imports=fix_imports, encoding=encoding, errors=errors)
    except TypeError:
        warnings.warn(
            "presenting '{!r}' instead of stored lambda 'type'".format(
                _moc_numpy_array_object_lambda
            ),
            MockedLambdaWarning
        )
        return _moc_numpy_array_object_lambda
