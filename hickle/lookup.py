"""
#lookup.py

This file manages all the mappings between hickle/HDF5 metadata and python
types.
There are three dictionaries that are populated here:

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

1) Create a file called load_[newstuff].py in loaders/
2) In the load_[newstuff].py file, define your create_dataset and load_dataset
   functions, along with the 'class_register' and 'exclude_register' lists.

"""


# %% IMPORTS
# Built-in imports
import sys
import warnings
import types
import io
import operator
import functools as ft
import weakref
from importlib.util import find_spec, module_from_spec

# Package imports
import dill as pickle
import copyreg
import numpy as np
import h5py as h5

# hickle imports
from .helpers import PyContainer,not_dumpable,nobody_is_my_name,no_compression


# %% GLOBALS
# Define dict of all acceptable types
types_dict = {}

# Define dict of all acceptable hickle types
hkl_types_dict = {}

# Define dict of all acceptable hickle container types
hkl_container_dict = {}

# Empty list (hashable) of loaded loader names
loaded_loaders = set()

# %% FUNCTION DEFINITIONS


def load_nothing(h_node,base_type,py_obj_type): # pragma: nocover
    """
    loads nothing
    """
    return nobody_is_my_name

def dump_nothing(py_obj, h_group, name, **kwargs): # pragma: nocover
    """
    dumps nothing
    """
    return nobody_is_my_name

# h5py < 2.10 ref_dtype has to be explicitly created by
# call to special_dtype(ref=h5py.Reference) 
# h5py >= 2.10 provides already a ref_dtype where metadata is already properly
# set. Both ways to define ref_dtype are equivalent for h5py >= 2.10
# for any other mimick definiton for ref_dtype of h5py >= 2.10
h5_version = [int(v) for v in h5.__version__.split('.')]
if h5_version[0] > 2 or ( h5_version[0] == 2 and h5_version[1] >= 10 ):
    link_dtype = h5.ref_dtype
else: # pragma: nocover
    link_dtype = np.dtype('O',metadata = {'ref':h5.Reference})


# %% CLASS DEFINITIONS

class _DictItem(): # pragma: nocover
    """
    dummy py_obj for dict_item loader
    """

class NodeReference(): # pragma: nocover
    """
    dummy py_obj_type returned by ReferenceManager.get_type
    when encountering dataset of h5py.ref_dtype which like 
    datasets containing pickled object instances have no explicit
    'type' attribute.
    """

class ReferenceError(Exception): # pragma: nocover
    """
    exception thrown by ReferenceManager
    """

class LookupError(Exception): # pragma: nocover
    """
    exception thrown if type lookup fails
    """
    

class SerializedWarning(UserWarning): # pragma: nocover
    """ An object type was not understood

    The data will be serialized using pickle.
    """

class MockedLambdaWarning(UserWarning): # pragma: nocover
    """ In Python >= 3.8 lambda function fails pickle.loads

    faking restoring lambda to keep hickle 4.0.X files
    loadin properly
    """

class ManagerMeta(type):
    """
    Metaclas for all manager classes derived from the BaseManager class.
    Ensures that the __managers__ class attribute of each immediate child
    class of BaseManager is intialized to a dictionary and that the class 
    definition none of its child overwritie its __managers__ attribute
    """

    def __new__(cls,name,bases,namespace,**kwords):
        if not any( isinstance(getattr(base,'__managers__',None),dict) for base in bases ):
            if not isinstance(namespace.get('__managers__',None),dict):
                namespace['__managers__'] = dict() if bases and not object in bases else None
        else:
            namespace.pop('__managers__',None)
        return super(ManagerMeta,cls).__new__(cls,name,bases,namespace,**kwords)
                
class BaseManager(metaclass = ManagerMeta):
    """
    Base class providing basic Management of simultaneously open managers.
    Must be subclassed.
    """
    __slots__ = ('__weakref__',)

    __managers__ = None

    @classmethod
    def _drop_manager(cls,fileid):
        """
        finalizer callback to properly remove a <manager> object from
        the <manager>.__managers__ structure when corresponding file is
        closed or ReferenceManager object is garbage collected

        Parameters:
        -----------
            cls (BaseManager):
                the child <manager> class for wich to drop the instance referred
                to by the specified file id
            fileid (h5py.FileId):
                Id identifying the hdf5 file the ReferenceManager was created for

        """
        try:
            manager = cls.__managers__.get(fileid,None)
            if manager is None:
                return
            cls.__managers__.pop(fileid,None)
        except: # pragma: nocover
            pass

    @classmethod
    def create_manager(cls,h_node,create_entry):#h_node,pickle_loads = pickle.loads):
        """
        Check whether an istance already exists for the file h_node belongs to and
        calls create_entry if no entry yet exists.

        Parameters:
        -----------
            cls:
                the manager class to create a new instance for

            h_node (h5py.File, h5py.Group, h5py.Dataset):
                the h5py node or its h_root_group to create a new ReferenceManager
                object for.

            create_entry (callable):
                function or method which returns a new table entry. A table entry
                is a tuple or list with contains as it first item the newly
                created <manager> object. It may include further items specific to
                the actual subclass.
                
                
        """
        manager = cls.__managers__.get(h_node.file.id,None)
        if manager is not None:
            raise ReferenceError("'{}' type manager already created for file '{}'".format(cls.__name__,h_node.file.filename))
        #root = ReferenceManager.get_root(h_node)
        table = cls.__managers__[h_node.file.id] = create_entry()
        weakref.finalize(table[0],cls._drop_manager,h_node.file.id)
        return table[0]

    def __init__(self):
        if type.mro(self.__class__)[0] is BaseManager:
            raise TypeError("'BaseManager' class must be subclassed")

    def __enter__(self):
        raise NotImplementedError("'{}' type object must implement ContextManager protocol")

    def __exit__(self,exc_type,exc_value,exc_traceback,h_node=None):
        # remove this ReferenceManager object from the table of active ReferenceManager objects
        # and cleanly unlink from any h5py object instance and id references managed. Finalize
        # 'hickle_types_table' overlay if it was created by __init__ for hickle 4.0.X file
        self.__class__._drop_manager(h_node.file.id)

class ReferenceManager(BaseManager,dict):
    """
    Manages all object and type references created for basic
    and type special memoisation

    To create a ReferenceManager call ReferenceManager.create_manager
    function. The value returned can be ans shall be used within a
    with statement for example as follows:

        with ReferenceManager.create_manager(h_root_group) as memo:
            _dump(data,h_root_group,'data',memo,**kwargs)

        with ReferenceManager.create_manager(h_root_group) as memo:
            _load(py_container,'data',h_root_group['data'],memo,load_loader = load_loader)

        with ReferenceManager.create_manager(h_root_group,fix_lambda_obj_type) as memo:
            _load(py_container,'data',h_root_group['data'],memo,load_loader = load_legacy_loader)

    """

    __slots__ = ('_py_obj_type_table','_py_obj_type_link','_base_type_link','_overlay','pickle_loads')

    
    @staticmethod
    def get_root(h_node):
        """
        returns the h_root_group the passed h_node belongs to.
        """

        # try to resolve the 'type' attribute of the h_node
        entry_ref = h_node.attrs.get('type',None)
        if isinstance(entry_ref,h5.Reference):
            # return the grand parent of the reffered to py_obj_type dataset as it
            # ist also the h_root_group of h_node
            entry = h_node.file.get(entry_ref,None)
            if entry is not None:
                return entry.parent.parent
        elif h_node.parent == h_node.file:
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
            # 'type' seems to be a a byte string or string fallback to h_node.file
            return h_node.file
        try:
            entry = h_node.parent.get(entry_ref,None)
        except ValueError:
            entry = None
        if entry is None:
            # 'type' reference seemst to be stale
            return h_node if isinstance(h_node,h5.Group) else h_node.file
        # return the grand parent of the reffered to py_obj_type dataset as it
        # ist also the h_root_group of h_node
        return  entry.parent.parent


        
    @staticmethod
    def _drop_overlay(h5file):
        """
        closes in memory overlay file providing dummy 'hickle_types_table' structure
        for hdf5 files which were created by hickle V4.0.X. 
        """
        h5file.close()
        #overlay.close()

    @classmethod
    def create_manager(cls,h_node,pickle_loads = pickle.loads):
        """
        creates an new ReferenceManager object for the h_root_group the h_node
        belongs to. 


        Parameters:
        -----------
            h_node (h5py.Group, h5py.Dataset):
                the h5py node or its h_root_group to create a new ReferenceManager
                object for.
            pickle_loads (FunctionType,MethodType):
                method to be used to expand py_obj_type pickle strings.
                defaults to pickle.loads. Must be set to fix_lambda_obj_type 
                for hickle created by hickle 4.0.x.

        Raises:
        -------
            ReferenceError:
                ReferenceManager has already been created for h_node or its h_root_group 
        """
        root = ReferenceManager.get_root(h_node)
        def create_manager():
            return (ReferenceManager(h_node,pickle_loads = pickle_loads),ReferenceManager.get_root(h_node))
        return super().create_manager(h_node,create_manager)

    def __init__(self,h_root_group,*args,pickle_loads = pickle.loads,**kwargs):
        """
        constructs ReferenceManager object

        Parameters:
        -----------
            h_root_group (h5py.Group):
                see ReferenceManager.create_manager

            args (tuple,list):
                passed to dict.__init__

            pickle_loads (FunctionType,MethodType):
                see ReferenceManager.create_manager

            kwargs (dict):
                passed to dict.__init__

        Raises:
        -------
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
        # an empty dummy hickle_types_table in order to ensure ReferenceManage.resolve_type
        # funkion still works properly
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

        # if h_root_group.file was opened for writing restore '_py_obj_type_link' and '_base_type_link'
        # table entries from '_py_obj_type_table' to ensure when h5py.Group and h5py.Dataset are added
        # anew to h_root_group tree structure their 'type' attribute is set to the correct py_obj_type
        # refrence by the ReferenceManager.store_type method. Each of '_py_obj_type_link' and '_base_type_link'
        # tables can be used to properly restore  the 'py_obj_type' and 'base_type' when loading the file
        # as well as assining to the 'type' attribute the appropiate 'py_obj_type' dataset reference from
        # the '_py_obj_type_table' when dumping data to the file.
        if h_root_group.file.mode != 'r+':
            return
        for index,entry in self._py_obj_type_table.items():
            if entry.shape is None and entry.dtype == 'S1':
                base_type = entry.name.rsplit('/',1)[-1].encode('ascii')
                self._base_type_link[base_type] = entry
                self._base_type_link[entry.id] = base_type
                continue
            py_obj_type = pickle.loads(entry[()])
            base_type_ref = entry.attrs.get('base_type',None)
            if not isinstance(base_type_ref,h5.Reference):
                raise ReferenceError("inconsistent 'hickle_types_table' entryies for py_obj_type '{}': no base_type".format(py_obj_type))
            try:
                base_type_entry = entry.file.get(base_type_ref,None)
            except ValueError:
                raise ReferenceError("inconsistent 'hickle_types_table' entryies for py_obj_type '{}': stale base_type".format(py_obj_type))
            base_type = self._base_type_link.get(base_type_entry.id,None)
            if base_type is None:
                base_type = base_type_entry.name.rsplit('/',1)[-1].encode('ascii')
            py_obj_type_id = id(py_obj_type)
            self._py_obj_type_link[py_obj_type_id] = entry
            self._py_obj_type_link[entry.id] = (py_obj_type_id,base_type)

    def store_type(self,h_node, py_obj_type, base_type = None,**kwargs):
        """
        assings a 'py_obj_type' entry reference to the 'type' attribute
        of h_node and creates if not present the appropriate 'hickle_types_table'
        entries for py_obj_type and base_type.

        Note:
        -----
            Storing and restoring the content of nodes contianing pickle byte strings
            is fully managed by pickle.dumps and pickle.loads functions including
            selection of appropiate py_obj_type. Therefore no explicit entry for 
            object and b'pickle' py_obj_type and base_type pairs indicating pickled
            content of pickled dataset are created.

        Parameters:
        -----------
            h_node (h5py.Group, h5py.Dataset):
                node the 'type' attribute shall be assiged to 'hickle_types_table'
                entry corresponding to provided py_obj_type, base_type entry pair.

            py_obj_type (any type or class):
                the type or class of the object instance represented by h_node

            base_type (bytes):
                the basetype bytes string of the loader used to create the h_node and
                restore an object instance form on load. If None no 'hickle_types_table'
                will be crated for py_obj_type if not already present and a LookupError
                exception is raised instead.

            kwargs (dict):
                keyword arguments to be passed to h5py.Group.create_dataset function
                when creating the entries for py_obj_type and base_type anew

        Raises:
        -------
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
        # futher use by ReferenceManager.store_type and ReferenceManager.resolve_type
        # methods
        py_obj_type_id = id(py_obj_type)
        entry = self._py_obj_type_link.get(py_obj_type_id,None)
        if entry is None:
            if base_type is None:
                raise LookupError("no entry found for py_obj_type '{}'".format(py_obj_type.__name__)) 
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
            # assing a reference to base_type entry within 'hickle_types_table' to
            # the 'base_type' attribute of the newly created py_obj_type entry.
            # if no 'hickle_types_table' does not yet contain empty dataset entry for
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
        h_node.attrs['type'] = entry.ref

    def resolve_type(self,h_node):
        """
        resolves the py_obj_type and base_type pair referred to by the 'type' attribute and
        if present the 'base_type' attribute.

        Note: If the 'base_type' is present it is assumed that the dataset was created by
            hickle 4.0.X version. Consequently it is assumed that the 'type' attribute
            to contains a pickle bytes string to load the py_obj_type from instead of a 
            reference a 'hickle_types_table' entry representing the py_obj_type base_type pair
            of h_node.

        Note: If 'type' attribute is not present h_node represents either a h5py.Reference to
            the actual node of the object to be restored or contains a pickle bytes string.
            Either  cases the implicit py_obj_type base_type pair beeing
            (NodeReference,b'!node-reference') and (object,b'pickle') respective is assumed.

        Parameters:
        -----------
            h_node (h5py.Group,h5py.Dataset):
                the node to resolve py_obj_type and base_type for

        Returns:
            tuple containing (py_obj_type,base_type,is_container)

            py_obj_type:
                the python type of the restored object

            base_type:
                the base_type string indicating the loader to be used for properly
                restoring the py_obj_type instance

            is_container:
                booling flag indicating whether h_node represents a h5py.Group or
                h5py.Reference both of which have to be handled by corresponding
                PyContainer type loaders or a h5py.Dataset for which the appropriate
                load_fn is to be called.
        """

        # load the 'type' type attribute. If not present check if h_node
        # is h5py.Reference dataset or a dataset containing a pickle bytes
        # string. In either case assume (NodeReference,b'!node-reference!') and (object,b'pickle')
        # respective as (py_obj_type, base_type) pair and set is_container flag to True for
        # h5py.Reference and False otherwise.
        # 
        # NOTE: hickle 4.0.X legacy file does not store 'type' attribute for h5py.Group nodes with 
        #       b'dict_item' base_type. As h5py.Groups do not have a dtype attribute the 
        #       check if h_node.dtype equals link_dtype/ref_dtype will raise AttributeError.
        #       If h_node represents a b'dict_item' than self.pickle_loads will point to
        #       fix_lambda_obj_type below which will properly handle None value of type_ref
        #       in any other case file is not a hickle 4.0.X legacy file and thus has to be
        #       considered broken
        type_ref = h_node.attrs.get('type',None)
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

            # check if 'type' attribute of h_node contains a reference to a 'hickle_types_table' entry
            # If not use pickle to restore py_object_type from the 'type' attribute value directly if possible
            try:
                # set is_container_flag to True if h_node is h5py.Group type object and false otherwise
                return self.pickle_loads(type_ref),h_node.attrs.get('base_type',b'pickle'),isinstance(h_node,h5.Group)
            except (TypeError,pickle.UnpicklingError,EOFError):
                raise ReferenceError("node '{}': 'type' attribute ('{}')invalid: not a pickle byte string".format(h_node.name,type_ref))
        try:
            entry = self._py_obj_type_table[type_ref]
        except (ValueError,KeyError):
            raise ReferenceError("node '{}': 'type' attribute invalid: stale reference")

        # load (py_obj_type,base_type) pair from _py_obj_type_link for resolve
        # 'hickle_types_table' entry referred to by 'type' entry
        # create appropriate _py_obj_type_link and _base_type_link entries if
        # (py_obj_type,base_type) pair for further use by ReferenceManager.store_type
        # and ReferenceManager.resolve_type methods.
        type_info = self._py_obj_type_link.get(entry.id,None)
        if type_info is None:
            base_type_ref = entry.attrs.get('base_type',None)
            if base_type_ref is None:
                base_type = b'pickle'
            else:
                try:
                    base_type_entry = self._py_obj_type_table[base_type_ref]
                except ( ValueError,KeyError ):
                    raise ReferenceError("stale base_type reference encoutnered for '{}' type table entry".format(entry.name))
                base_type = self._base_type_link.get(base_type_entry.id,None)
                if base_type is None:
                    # get the relativetable entry name form full path name of entry node
                    base_type = base_type_entry.name.rsplit('/',1)[-1].encode('ASCII')
                    self._base_type_link[base_type] = base_type_entry
                    self._base_type_link[base_type_entry.id] = base_type
            py_obj_type = self.pickle_loads(entry[()])
            self._py_obj_type_link[id(py_obj_type)] = entry
            type_info = self._py_obj_type_link[entry.id] = (py_obj_type,base_type)
        # return (py_obj_type,base_type). set is_container flag to true if 
        # h_node is h5py.Group object and false otherwise
        return (*type_info,isinstance(h_node,h5.Group))

    def __enter__(self):
        if not isinstance(self._py_obj_type_table,h5.Group) or not self._py_obj_type_table:
            raise RuntimeError("Stale ReferenceManager, call ReferenceManager.create_manager to create a new one")
        return self

    def __exit__(self,exc_type,exc_value,exc_traceback):
        if not isinstance(self._py_obj_type_table,h5.Group) or not self._py_obj_type_table:
            return
        # remove this ReferenceManager object from the table of active ReferenceManager objects
        # and cleanly unlink from any h5py object instance and id references managed. Finalize
        # 'hickle_types_table' overlay if it was created by __init__ for hickle 4.0.X file
        super().__exit__(exc_type,exc_value,exc_traceback,self._py_obj_type_table)
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

_managed_by_hickle = {'hickle',''}
# This function registers a class to be used by hickle
def register_class(myclass_type, hkl_str, dump_function=None, load_function=None, container_class=None,memoise = True):
    """ Register a new hickle class.

    Parameters:
    -----------
        myclass_type type(class): type of class
        hkl_str (str): String to write to HDF5 file to describe class
        dump_function (function def): function to write data to HDF5
        load_function (function def): function to load data from HDF5
        container_class (class def): proxy class to load data from HDF5
        memoise (bool): 
            True: references to the object instances shall be remembered
                during dump and load for properly resolving multiple
                references to the same object instance.
            False: every occurence of an instance of the object has to be dumped
                and restored on load disregarding instances already present.

    Raises:
    -------
        TypeError:
            myclass_type represents a py_object the loader for which is to
            be provided by hickle.lookup and hickle.hickle module only
            
    """

    if (
        myclass_type is object or
        isinstance(
            myclass_type,
            (types.FunctionType,types.BuiltinFunctionType,types.MethodType,types.BuiltinMethodType)
        ) or
        issubclass(myclass_type,(type,_DictItem))
    ):
        # object as well als all kinds of functions and methods as well as all class objects and
        # the special _DictItem class are to be handled by hickle core only. 
        dump_module = getattr(dump_function,'__module__','').split('.',2)
        load_module = getattr(load_function,'__module__','').split('.',2)
        container_module = getattr(container_class,'__module__','').split('.',2)
        if {dump_module[0],load_module[0],container_module[0]} - _managed_by_hickle:
            raise TypeError(
                "loader for '{}' type managed by hickle only".format(
                    myclass_type.__name__
                )
            )
        if "loaders" in {*dump_module[1:2],*load_module[1:2],*container_module[1:2]}:
            raise TypeError(
                "loader for '{}' type managed by hickle core only".format(
                    myclass_type.__name__
                )
            )
    # add loader
    if dump_function is not None:
        types_dict[myclass_type] = ( dump_function, hkl_str,memoise)
    if load_function is not None:
        hkl_types_dict[hkl_str] = load_function
    if container_class is not None:
        hkl_container_dict[hkl_str] = container_class


def register_class_exclude(hkl_str_to_ignore):
    """ Tell loading funciton to ignore any HDF5 dataset with attribute
    'type=XYZ'

    Args:
        hkl_str_to_ignore (str): attribute type=string to ignore and exclude
            from loading.
    """

    if hkl_str_to_ignore in disallowed_to_ignore:
        raise ValueError(
            "excluding '{}' base_type managed by hickle core not possible".format(
                hkl_str_to_ignore
            )
        )
    hkl_types_dict[hkl_str_to_ignore] = load_nothing
    hkl_container_dict[hkl_str_to_ignore] = NoContainer


def load_loader(py_obj_type, type_mro = type.mro):
    """
    Checks if given `py_obj` requires an additional loader to be handled
    properly and loads it if so. 

    Parameters:
    -----------
        py_obj:
            the Python object to find an appropriate loader for

    Returns:
    --------
        py_obj:
            the Python object the loader was requested for

        (create_dataset,base_type,memoise):
            tuple providing create_dataset function, name of base_type
            used to represent py_obj and the boolean memoise flag 
            indicating whether loaded object shall be remembered
            for restoring further references to it or must be loaded every time
            encountered.

    Raises:
    -------
        RuntimeError:
            in case py object is defined by hickle core machinery.

    """

    # any function or method object, any class object will be passed to pickle
    # ensure that in any case create_pickled_dataset is called.

    # get the class type of py_obj and loop over the entire mro_list
    for mro_item in type_mro(py_obj_type):
        # Check if mro_item can be found in types_dict and return if so
        loader_item = types_dict.get(mro_item,None)
        if loader_item is not None:
            return py_obj_type,loader_item

        # Obtain the package name of mro_item
        package_list = mro_item.__module__.split('.',2)

        if package_list[0] == 'hickle':
            if package_list[1] != 'loaders':
                print(mro_item,package_list)
                raise RuntimeError(
                    "objects defined by hickle core must be registerd"
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
            # Obtain the name of the associated loader
            loader_name = 'hickle.loaders.load_{:s}'.format(package_list[0])

            # Check if this module is already loaded, and return if so
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
    
                # no module sepecification found for module
                # check next base class
                continue
            # import the the loader module described by module_spec
            # any import errors and exceptions result at this stage from
            # errors inside module and not cause loader module does not
            # exists
            loader = module_from_spec(loader_spec)
            loader_spec.loader.exec_module(loader)
            sys.modules[loader_name] = loader

        # load all loaders defined by loader module
        # no performance benefit of starmap or map if required to build
        # list or tuple of None's returned
        for next_loader in loader.class_register:
            register_class(*next_loader)
        for drop_loader in loader.exclude_register:
            register_class_exclude(drop_loader)
        loaded_loaders.add(loader_name)

        # check if loader module defines a loader for base_class mro_item
        loader_item = types_dict.get(mro_item,None)
        if loader_item is not None:
            # return loader for base_class mro_item
            return py_obj_type,loader_item
        # the new loader does not define loader for mro_item
        # check next base class

    # no appropriate loader found return fallback to pickle
    return py_obj_type,(create_pickled_dataset,b'pickle',True)

def type_legacy_mro(cls):
    """
    drop in replacement of type.mro for loading legacy hickle 4.0.x files which were
    created without generalized PyContainer objects in mind. consequently some
    h5py.Datasets and h5py.Group objects expose function objets as their py_obj_type
    type.mro expects classes only.

    Parameters:
    -----------
        cls (type):
            the py_obj_type/class of the object to load or dump

    Returns:
    --------
        mro list for cls as returned by type.mro  or in case cls is a function or method
        a single element tuple is returned
    """
    if isinstance(
        cls,
        (types.FunctionType,types.BuiltinFunctionType,types.MethodType,types.BuiltinMethodType)
    ):
        return (cls,)
    return type.mro(cls) 

load_legacy_loader = ft.partial(load_loader,type_mro = type_legacy_mro)

# %% BUILTIN LOADERS (not maskable)

# list of below hkl_types which may not be ignored
disallowed_to_ignore = {b'dict_item',b'pickle',b'!node-reference!',b'moc_lambda'}

class NoContainer(PyContainer): # pragma: nocover
    """
    load nothing container
    """

    def convert(self):
        pass


class _DictItemContainer(PyContainer):
    """
    PyContainer reducing hickle version 4.0.0 dict_item type h5py.Group to 
    its content for inclusion within dict h5py.Group
    """

    def convert(self):
        return self._content[0]

register_class(_DictItem, b'dict_item',dump_nothing,load_nothing,_DictItemContainer)

        
class ExpandReferenceContainer(PyContainer):
    """
    PyContainer for properly restoring addtional references
    to an object instance shared multiple times within the dumped
    object structure
    """

    def filter(self,h_parent):
        """
        resolves the h5py.Reference link and yields the
        the node it refers to as subitem of h_parent so
        that it can be porperly loaded by recursively loading
        hickle._load method independent whether it can be directly
        be loaded from the memo dictionary or has to be
        restored from file. 
        """
        try:
            referred_node = h_parent.file[h_parent[()]]
        except ( ValueError, KeyError ):
            raise ReferenceError("node '{}' stale node reference".format(h_parent.name))
        yield referred_node.name.rsplit('/',1)[-1], referred_node

    def convert(self):
        """
        returns the object the reference was pointing to
        """
        return self._content[0]

# objects created by resolving h5py.Reference datasets are already stored inside
# memo dictionary so no need to remoise them.
register_class(NodeReference,b'!node-reference!',dump_nothing,load_nothing,ExpandReferenceContainer,False)


def create_pickled_dataset(py_obj, h_group, name, reason = None, **kwargs):
    """
    Create pickle string as object can not be mapped to any other h5py
    structure. In case raise a warning and convert to pickle string.

    Args:
        py_obj: python object to dump; default if item is not matched.
        h_group (h5.File.group): group to dump data into.
        name (str): the name of the resulting dataset
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
    d = h_group.create_dataset(name, data=memoryview(pickled_obj), **kwargs)
    return d,() 

def load_pickled_data(h_node, base_type, py_obj_type):
    """
    loade pickle string and return resulting py_obj
    """
    return pickle.loads(h_node[()])

        
# no dump method is registered for object as this is the default for
# any unknown object and for classes, functions and methods
register_class(object,b'pickle',None,load_pickled_data)


def _moc_numpy_array_object_lambda(x):
    """
    drop in replacement for lambda object types which seem not
    any more be accepted by pickle for Python 3.8 and onward.
    see fix_lambda_obj_type function below

    Parameters:
    -----------
        x (list): itemlist from which to return first element

    Returns:
        first element of provided list
    """
    return x[0]

register_class(_moc_numpy_array_object_lambda,b'moc_lambda',dump_nothing,load_nothing)

def fix_lambda_obj_type(bytes_object, *, fix_imports=True, encoding="ASCII", errors="strict"):
    """
    drop in replacement for pickle.loads method when loading files created by hickle 4.0.x 
    It captures any TypeError thrown by pickle.loads when encountering a picle string 
    representing a lambda function used as py_obj_type for a h5py.Dataset or h5py.Group
    While in Python <3.8 pickle loads creates the lambda Python >= 3.8 throws an 
    error when encountering such a pickle string. This is captured and _moc_numpy_array_object_lambda
    returned instead. futher some h5py.Group and h5py.Datasets do not provide any 
    py_obj_type for them object is returned assuming that proper loader has been identified
    by other objects already
    """
    if bytes_object is None:
        return object
    try:
        return pickle.loads(bytes_object,fix_imports=fix_imports,encoding=encoding,errors=errors)
    except TypeError:
        warnings.warn(
            "presenting '{!r}' instead of stored lambda 'type'".format(
                _moc_numpy_array_object_lambda
            ),
            MockedLambdaWarning
        )
        return _moc_numpy_array_object_lambda
