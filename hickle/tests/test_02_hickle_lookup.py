#! /usr/bin/env python
# encoding: utf-8
"""
# test_hickle_lookup.py

Unit tests for hickle module -- lookup functions.

"""

# %% IMPORTS
import pytest
import sys
import shutil
import types
import weakref
import compileall
import os

# Package imports
import re
import collections
import numpy as np
import h5py
import dill as pickle
from importlib.util import find_spec,spec_from_loader,spec_from_file_location
from importlib import reload
from copy import copy
import os.path
from py.path import local

# hickle imports
from hickle.helpers import PyContainer,not_dumpable
from hickle.loaders import optional_loaders, attribute_prefix
import hickle.lookup as lookup
    
# Set current working directory to the temporary directory
local.get_temproot().chdir()

# %% DATA DEFINITIONS

dummy_data = (1,2,3)


# %% FIXTURES

@pytest.fixture
def h5_data(request):
    """
    create dummy hdf5 test data file for testing PyContainer, H5NodeFilterProxy and
    ReferenceManager. Uses name of executed test as part of filename
    """

    dummy_file = h5py.File('hickle_lookup_{}.hdf5'.format(request.function.__name__),'w')
    filename = dummy_file.filename
    test_data = dummy_file.create_group("root_group")
    yield test_data
    dummy_file.close()

@pytest.fixture()
def loader_table():

    """
    create a class_register and a exclude_register table for testing
    register_class and register_class_exclude functions

    0: dataset only loader
    1: PyContainer only loader
    2: not dumped loader
    3: external loader module trying to overwrite hickle core loader
    4: hickle loader module trying to overload hickle core loader
    3: loader defined by hickle core
    
    """

    # clear loaded_loaders, types_dict, hkl_types_dict and hkl_contianer_dict
    # to ensure no loader preset by hickle core or hickle loader module
    # intervenes with test
    global lookup
    lookup.LoaderManager.__loaded_loaders__.clear()
    tuple( True for opt in lookup.LoaderManager.__py_types__.values() if opt.clear() )
    tuple( True for opt in lookup.LoaderManager.__hkl_functions__.values() if opt.clear() )
    tuple( True for opt in lookup.LoaderManager.__hkl_container__.values() if opt.clear() )

    # simulate loader definitions found within loader modules
    def create_test_dataset(myclass_type,h_group,name,**kwargs):
        return h_group,()

    def load_test_dataset(h_node,base_type,py_obj_type):
        return 12

    class TestContainer(PyContainer):
        def convert(self):
            return self._content[0]

    class NotHicklePackage(TestContainer):
        """
        checks if container_class provided by module outside 
        hickle package  tries to define alternative loader for
        IteratorProxy class handled by hickle core directly
        """
        __module__ = "nothickle.loaders.load_builtins"

    class HickleLoadersModule(TestContainer):
        """
        checks if container_class provided by 
        hickle.loaders module tries to define alternative loader for
        IteratorProxy class handled by hickle core directly
        """
        __module__ = "hickle.loaders.load_builtins"

    class IsHickleCore(TestContainer):
        """
        Simulates loader registered by hickle.hickle module
        """
        __module__ = "hickle.hickle"

    # provide the table
    yield [
        (int,b'int',create_test_dataset,load_test_dataset,None,False),
        (list,b'list',create_test_dataset,None,TestContainer,True),
        (tuple,b'tuple',None,load_test_dataset,TestContainer),
        (lookup._DictItem,b'dict_item',None,None,NotHicklePackage),
        (lookup._DictItem,b'pickle',None,None,HickleLoadersModule),
        (lookup._DictItem,b'dict_item',lookup.LoaderManager.register_class,None,IsHickleCore)
    ]

    # cleanup and reload hickle.lookup module to reset it to its initial state
    # in case hickle.hickle has already been preloaded by pytest also reload it
    # to ensure no side effects occur during later tests
    lookup.LoaderManager.__loaded_loaders__.clear()
    tuple( True for opt in lookup.LoaderManager.__py_types__.values() if opt.clear() )
    tuple( True for opt in lookup.LoaderManager.__hkl_functions__.values() if opt.clear() )
    tuple( True for opt in lookup.LoaderManager.__hkl_container__.values() if opt.clear() )
    
    reload(lookup)
    lookup = sys.modules[lookup.__name__]
    hickle_hickle = sys.modules.get("hickle.hickle",None)
    if hickle_hickle is not None:
        reload(hickle_hickle)

# %% CLASS DEFINITIONS

class ToBeInLoadersOrNotToBe():
    """
    Dummy class used to check that only loaders for Python objects
    are accepted by load_loader which are either declared
    outside hickle or are pre registered by hickle core through directly
    calling register_class or are declared by a load_<module>.py module
    within the pickle.loaders package

    Also it is used in simulating reduced object tuple with all trailing
    None items removed
    """
    __slots__ = ()

    def __reduce_ex__(self,proto = pickle.DEFAULT_PROTOCOL):
        reduced = super(ToBeInLoadersOrNotToBe,self).__reduce_ex__(proto)
        for index,item in enumerate(reduced[:1:-1],0):
            if item is not None:
                return reduced[:(-index if index > 0 else None)]
        return reduced

    def __reduce__(self):
        reduced = super(ToBeInLoadersOrNotToBe,self).__reduce__()
        for index,item in enumerate(reduced[:1:-1],0):
            if item is not None:
                return reduced[:(-index if index > 0 else None)]
        return reduced

    def __eq__(self,other):
        return other.__class__ is self.__class__

    def __ne__(self,other):
        return self != other
class ClassToDump():
    """
    Primary class used to test create_pickled_dataset function
    """
    def __init__(self,hallo,welt,with_default=1):
        self._data = hallo,welt,with_default

    def __eq__(self,other):
        return other.__class__ is self.__class__ and self._data == other._data

    def __ne__(self,other):
        return self != other

class ClassToDumpCompact(ClassToDump):
    """
    Class which may be handled by 'compact_expand' loader
    """
    def __compact__(self):
        return self._data

    def __expand__(self,compact):
        self._data = compact

class ClassToDumpCompactOff(ClassToDump):
    """
    Class which enforces that any instance is pickled
    independent whether 'compact_expand' loader was selected
    for hickle.dump call or not
    """
    def __compact__(self):
        return None

class ClassToDumpCompactStrange(ClassToDump):
    """
    Class which does not properly implement
    '__compact__' and '__expand__' methods
    recommended by compact expand protocol
    """
    def __compact__(self):
        return self._data

class ClassToDumpCompactStrange2(ClassToDump):
    """
    Another class which does not properly implement
    '__compact__' and '__expand__' methods
    recommended by compact expand protocol
    """
    def __compact__(self):
        return 42
    
class ClassToDumpCompactDataset(ClassToDump):
    """
    Class which is to be represented by a h5py.Dataset
    instead of a h5py.Group in its compacted form
    """
    def __compact__(self):
        return "{}|{}|{}".format(*self._data)

    def __expand__(self,compact):
        self._data = compact.split("|")
        self._data[2] = int(self._data[2])
        self._data = (*self._data,)

class SimpleClass():
    """
    simple class used to check that instance __dict__ is properly dumped and
    restored by create_pickled_dataset and PickledContainer
    """
    def __init__(self):
        self.someattr = "I'm some attr"
        self.someother = 12

    def __eq__(self,other):
        return other.__class__ is self.__class__ and self.__dict__ == other.__dict__

    def __ne__(self,other):
        return self != other

class NoExtendList(list):
    """
    special list class used to test whether append is properly used
    when list like object is dumped and restored through create_pickled_dataset
    and PickledContainer
    """

    def __getattribute__(self,name):
        if name == "extend":
            raise AttributeError("no extend")
        return super(NoExtendList,self).__getattribute__(name)

# %% FUNCTION DEFINITIONS

def function_to_dump(hallo,welt,with_default=1):
    """
    non class function to be dumped and restored through
    create_pickled_dataset and load_pickled_data
    """
    return hallo,welt,with_default

def test_AttemptRecoverCustom_classes(h5_data):
    recovered_group = lookup.RecoveredGroup({'hello':1},attrs={'world':2,'type':42})
    assert recovered_group == {'hello':1} and recovered_group.attrs == {'world':2}
    array_to_recover = np.random.random_sample([4,2])
    dataset_to_recover = h5_data.create_dataset('to_recover',data=array_to_recover)
    dataset_to_recover.attrs['world'] = 2
    dataset_to_recover.attrs['type'] = 42
    recovered_dataset = lookup.RecoveredDataset(dataset_to_recover[()],dtype=dataset_to_recover.dtype,attrs=dataset_to_recover.attrs)
    assert np.allclose(recovered_dataset,array_to_recover)
    assert recovered_dataset.dtype == array_to_recover.dtype
    assert recovered_dataset.attrs == {'world':2}
    #recovered = lookup.recover_custom_dataset(dataset_to_recover,'unknown',dataset_to_recover.attrs['type']) 
    #assert recovered.dtype == array_to_recover.dtype and recovered == array_to_recover
    #assert recovered.attrs == {'world':2}
    

def test_LoaderManager_register_class(loader_table):
    """
    tests the register_class method
    """

    # try to register dataset only loader specified by loader_table
    # and retrieve its contents from types_dict and hkl_types_dict
    loader_spec = loader_table[0]
    lookup.LoaderManager.register_class(*loader_spec)
    assert lookup.LoaderManager.__py_types__[None][loader_spec[0]] == (*loader_spec[2:0:-1],loader_spec[5])
    assert lookup.LoaderManager.__hkl_functions__[None][loader_spec[1]] == loader_spec[3]
    with pytest.raises(KeyError):
        lookup.LoaderManager.__hkl_container__[None][loader_spec[1]] is None

    # try to register PyContainer only loader specified by loader_table
    # and retrieve its contents from types_dict and hkl_container_dict
    loader_spec = loader_table[1]
    lookup.LoaderManager.register_class(*loader_spec)
    assert lookup.LoaderManager.__py_types__[None][loader_spec[0]] == (*loader_spec[2:0:-1],loader_spec[5])
    with pytest.raises(KeyError):
        lookup.LoaderManager.__hkl_functions__[None][loader_spec[1]] is None
    assert lookup.LoaderManager.__hkl_container__[None][loader_spec[1]] == loader_spec[4]


    # try to register container without dump_function specified by
    # loader table and try to retrieve load_function and PyContainer from
    # hkl_types_dict and hkl_container_dict
    loader_spec = loader_table[2]
    lookup.LoaderManager.register_class(*loader_spec)
    with pytest.raises(KeyError):
        lookup.LoaderManager.__py_types__[None][loader_spec[0]][1] == loader_spec[1]
    assert lookup.LoaderManager.__hkl_functions__[None][loader_spec[1]] == loader_spec[3]
    assert lookup.LoaderManager.__hkl_container__[None][loader_spec[1]] == loader_spec[4]

    # try to register loader shadowing loader preset by hickle core
    # defined by external loader module
    loader_spec = loader_table[3]
    with pytest.raises(TypeError,match = r"loader\s+for\s+'\w+'\s+type\s+managed\s+by\s+hickle\s+only"):
        lookup.LoaderManager.register_class(*loader_spec)
    loader_spec = loader_table[4]

    # try to register loader shadowing loader preset by hickle core
    # defined by hickle loaders module
    with pytest.raises(TypeError,match = r"loader\s+for\s+'\w+'\s+type\s+managed\s+by\s+hickle\s+core\s+only"):
        lookup.LoaderManager.register_class(*loader_spec)

    # simulate registering loader preset by hickle core
    loader_spec = loader_table[5]
    lookup.LoaderManager.register_class(*loader_spec)
    loader_spec = loader_table[0]
    lookup.LoaderManager.__hkl_functions__[None][b'!node-reference!'] = loader_spec[3:5]
    with pytest.raises(ValueError):
        lookup.LoaderManager.register_class(loader_spec[0],b'!node-reference!',*loader_spec[2:],'custom')
    lookup.LoaderManager.__hkl_functions__[None].pop(b'!node-reference!')
    with pytest.raises(lookup.LookupError):
        lookup.LoaderManager.register_class(*loader_spec,'mine')

def test_LoaderManager_register_class_exclude(loader_table):
    """
    test register class exclude function
    """

    # try to disable loading of loader preset by hickle core
    base_type = loader_table[5][1]
    lookup.LoaderManager.register_class(*loader_table[2])
    lookup.LoaderManager.register_class(*loader_table[5])
    with pytest.raises(ValueError,match = r"excluding\s+'.+'\s+base_type\s+managed\s+by\s+hickle\s+core\s+not\s+possible"):
        lookup.LoaderManager.register_class_exclude(base_type)

    # disable any of the other loaders
    base_type = loader_table[2][1]
    lookup.LoaderManager.register_class_exclude(base_type)
    with pytest.raises(lookup.LookupError):
        lookup.LoaderManager.register_class_exclude(base_type,'compact')


def patch_importlib_util_find_spec(name,package=None):
    """
    function used to temporarily redirect search for loaders
    to hickle_loader directory in test directory for testing
    loading of new loaders
    """
    return find_spec("hickle.tests." + name.replace('.','_',1),package)

def patch_importlib_util_find_spec_no_load_builtins(name,package=None):
    """
    function used to temporarily redirect search for loaders
    to hickle_loader directory in test directory for testing
    loading of new loaders
    """
    if name in {'hickle.loaders.load_builtins'}:
        return None
    return find_spec("hickle.tests." + name.replace('.','_',1),package)

def patch_importlib_util_spec_from_tests_loader(name, loader, *, origin=None, is_package=None):
    """
    function used to temporarily redirect search for loaders
    to hickle_loader directory in test directory for testing
    loading of new loaders
    """
    name = name.replace('.','_',1)
    myloader = copy(sys.modules['hickle.tests'].__loader__)
    myloader.name = "hickle.tests." + name
    myloader.path = os.path.join(os.path.dirname(myloader.path),'{}.py'.format(name))
    return spec_from_loader(myloader.name,myloader,origin=origin,is_package=is_package)

def patch_importlib_util_spec_from_loader(name, loader, *, origin=None, is_package=None):
    """
    function used to temporarily redirect search for loaders
    to hickle_loader directory in test directory for testing
    loading of new loaders
    """
    return spec_from_loader("hickle.tests." + name.replace('.','_',1),loader,origin=origin,is_package=is_package)

def patch_importlib_util_spec_from_file_location(name, location, *, loader=None, submodule_search_locations=None):
    """
    function used to temporarily redirect search for loaders
    to hickle_loader directory in test directory for testing
    loading of new loaders
    """
    return spec_from_file_location("hickle.tests." + name.replace('.','_',1),location,loader=loader,submodule_search_locations =submodule_search_locations)


def patch_importlib_util_find_no_spec(name,package=None):
    """
    function used to simulate situation where no appropriate loader 
    could be found for object
    """
    return None

def patch_importlib_util_no_spec_from_loader(name, loader, *, origin=None, is_package=None):
    """
    function used to simulate situation where no appropriate loader 
    could be found for object
    """
    return None

def patch_importlib_util_no_spec_from_file_location(name, location, *, loader=None, submodule_search_locations=None):
    """
    function used to simulate situation where no appropriate loader 
    could be found for object
    """
    return None
def patch_hide_collections_loader(name,package=None):
    if name in ('hickle.loaders.load_collections'):
        return None
    return find_spec(name,package)

def test_LoaderManager(loader_table,h5_data):
    """
    tests LoaderManager constructor
    """
    manager = lookup.LoaderManager(h5_data,False)
    assert isinstance(manager.types_dict,collections.ChainMap)
    assert manager.types_dict.maps[0] is lookup.LoaderManager.__py_types__[None]
    assert isinstance(manager.hkl_types_dict,collections.ChainMap)
    assert manager.hkl_types_dict.maps[0] is lookup.LoaderManager.__hkl_functions__[None]
    assert isinstance(manager.hkl_container_dict,collections.ChainMap)
    assert manager.hkl_container_dict.maps[0] is lookup.LoaderManager.__hkl_container__[None]
    assert manager._mro is type.mro
    assert manager._file.id == h5_data.file.id
    manager = lookup.LoaderManager(h5_data,True)
    assert manager.types_dict.maps[0] is lookup.LoaderManager.__py_types__['hickle-4.x']
    assert manager.types_dict.maps[1] is lookup.LoaderManager.__py_types__[None]
    assert manager.hkl_types_dict.maps[0] is lookup.LoaderManager.__hkl_functions__['hickle-4.x']
    assert manager.hkl_types_dict.maps[1] is lookup.LoaderManager.__hkl_functions__[None]
    assert manager.hkl_container_dict.maps[0] is lookup.LoaderManager.__hkl_container__['hickle-4.x']
    assert manager.hkl_container_dict.maps[1] is lookup.LoaderManager.__hkl_container__[None]
    assert manager._mro is lookup.type_legacy_mro
    assert manager._file.id == h5_data.file.id

    ###### amend #####
    manager = lookup.LoaderManager(h5_data,False,{'custom':True})
    assert manager.types_dict.maps[0] is lookup.LoaderManager.__py_types__['custom']
    assert manager.types_dict.maps[1] is lookup.LoaderManager.__py_types__[None]
    assert manager.hkl_types_dict.maps[0] is lookup.LoaderManager.__hkl_functions__['custom']
    assert manager.hkl_types_dict.maps[1] is lookup.LoaderManager.__hkl_functions__[None]
    assert manager.hkl_container_dict.maps[0] is lookup.LoaderManager.__hkl_container__['custom']
    assert manager.hkl_container_dict.maps[1] is lookup.LoaderManager.__hkl_container__[None]
    assert manager._file.id == h5_data.file.id
    assert h5_data.attrs.get('{}CUSTOM'.format(attribute_prefix),None)
    manager = lookup.LoaderManager(h5_data,False,None)
    assert manager.types_dict.maps[0] is lookup.LoaderManager.__py_types__['custom']
    assert manager.types_dict.maps[1] is lookup.LoaderManager.__py_types__[None]
    assert manager.hkl_types_dict.maps[0] is lookup.LoaderManager.__hkl_functions__['custom']
    assert manager.hkl_types_dict.maps[1] is lookup.LoaderManager.__hkl_functions__[None]
    assert manager.hkl_container_dict.maps[0] is lookup.LoaderManager.__hkl_container__['custom']
    assert manager.hkl_container_dict.maps[1] is lookup.LoaderManager.__hkl_container__[None]
    assert manager._file.id == h5_data.file.id
    h5_data.attrs.pop('{}CUSTOM'.format(attribute_prefix),None)
    manager = lookup.LoaderManager(h5_data,False,{'custom':False})
    assert manager.types_dict.maps[0] is lookup.LoaderManager.__py_types__[None]
    assert manager.hkl_types_dict.maps[0] is lookup.LoaderManager.__hkl_functions__[None]
    assert manager.hkl_container_dict.maps[0] is lookup.LoaderManager.__hkl_container__[None]
    assert h5_data.attrs.get('{}CUSTOM'.format(attribute_prefix),h5_data) is h5_data

    with pytest.raises(lookup.LookupError):
        manager = lookup.LoaderManager(h5_data,False,{'compact':True})


def test_LoaderManager_drop_manager(h5_data):
    """
    test static LoaderManager._drop_table method
    """
    loader = lookup.LoaderManager(h5_data)
    lookup.LoaderManager.__managers__[h5_data.file.id] = (loader,)
    some_other_file = h5py.File('someother.hdf5','w')
    some_other_root = some_other_file.create_group('root')
    lookup.LoaderManager._drop_manager(some_other_root.file.id)
    lookup.LoaderManager.__managers__[some_other_file.file.id] = (lookup.LoaderManager(some_other_root),)
    assert lookup.LoaderManager.__managers__.get(h5_data.file.id,None) == (loader,)
    lookup.LoaderManager._drop_manager(h5_data.file.id)
    assert lookup.LoaderManager.__managers__.get(h5_data.file.id,None) is None
    lookup.LoaderManager._drop_manager(some_other_root.file.id)
    assert not lookup.LoaderManager.__managers__
    some_other_file.close()
    
def test_LoaderManager_create_manager(h5_data):
    """
    test public static LoaderManager.create_manager function
    """
    second_tree = h5_data.file.create_group('seondary_root')
    loader = lookup.LoaderManager.create_manager(h5_data)
    assert lookup.LoaderManager.__managers__[h5_data.file.id][0] is loader
    with pytest.raises(lookup.LookupError):
        second_table = lookup.LoaderManager.create_manager(second_tree)
    lookup.LoaderManager._drop_manager(h5_data.file.id)

def test_LoaderManager_context(h5_data):
    """
    test use of LoaderManager as context manager
    """
    with lookup.LoaderManager.create_manager(h5_data) as loader:
        assert lookup.LoaderManager.__managers__[h5_data.file.id][0] is loader
    assert loader._file is None
    with pytest.raises(RuntimeError):
        with loader as loader2:
            pass
    loader.__exit__(None,None,None)

def test_LoaderManager_load_loader(loader_table,h5_data,monkeypatch):
    """
    test LoaderManager.load_loader method
    """

    # some data to check loader for
    # assume loader should be load_builtins loader
    py_object = dict()
    loader_name = "hickle.loaders.load_builtins"
    with monkeypatch.context() as moc_import_lib:
        with lookup.LoaderManager.create_manager(h5_data) as loader:

            # hide loader from hickle.lookup.loaded_loaders and check that 
            # fallback loader for python object is returned
            moc_import_lib.setattr("importlib.util.find_spec",patch_importlib_util_find_no_spec)
            moc_import_lib.setattr("hickle.lookup.find_spec",patch_importlib_util_find_no_spec)
            moc_import_lib.setattr("importlib.util.spec_from_loader",patch_importlib_util_no_spec_from_loader)
            moc_import_lib.setattr("hickle.lookup.spec_from_loader",patch_importlib_util_no_spec_from_loader)
            moc_import_lib.setattr("importlib.util.spec_from_file_location",patch_importlib_util_no_spec_from_file_location)
            moc_import_lib.setattr("hickle.lookup.spec_from_file_location",patch_importlib_util_no_spec_from_file_location)
            moc_import_lib.delitem(sys.modules,"hickle.loaders.load_builtins",raising=False)
            py_obj_type,nopickleloader = loader.load_loader(py_object.__class__)
            assert py_obj_type is dict and nopickleloader == (lookup.create_pickled_dataset,b'pickle',True)
            assert py_obj_type is dict and nopickleloader == (lookup.create_pickled_dataset,b'pickle',True)

            lookup._custom_loader_enabled_builtins[py_obj_type.__class__.__module__] = ('','')
            py_obj_type,nopickleloader = loader.load_loader(py_object.__class__)
            assert py_obj_type is dict and nopickleloader == (lookup.create_pickled_dataset,b'pickle',True)
            
            backup_builtins = sys.modules['builtins']
            moc_import_lib.delitem(sys.modules,'builtins')
            with pytest.warns(lookup.PackageImportDropped):# TODO when warning is added run check for warning
                py_obj_type,nopickleloader = loader.load_loader(py_object.__class__)
            assert py_obj_type is dict and nopickleloader == (lookup.create_pickled_dataset,b'pickle',True)
            moc_import_lib.setitem(sys.modules,'builtins',backup_builtins)


            # redirect load_builtins loader to tests/hickle_loader path
            moc_import_lib.setattr("importlib.util.spec_from_file_location",patch_importlib_util_spec_from_file_location)
            moc_import_lib.setattr("hickle.lookup.spec_from_file_location",patch_importlib_util_spec_from_file_location)
            #py_obj_type,nopickleloader = loader.load_loader(py_object.__class__)
            #assert py_obj_type is dict and nopickleloader == (lookup.create_pickled_dataset,b'pickle',True)
            moc_import_lib.setattr("importlib.util.find_spec",patch_importlib_util_find_spec)
            moc_import_lib.setattr("hickle.lookup.find_spec",patch_importlib_util_find_spec)
            moc_import_lib.setattr("importlib.util.spec_from_loader",patch_importlib_util_spec_from_loader)
            moc_import_lib.setattr("hickle.lookup.spec_from_loader",patch_importlib_util_spec_from_loader)
    
            # try to find appropriate loader for dict object, a mock of this
            # loader should be provided by hickle/tests/hickle_loaders/load_builtins
            # module ensure that this module is the one found by load_loader function
            import hickle.tests.hickle_loaders.load_builtins as load_builtins
            moc_import_lib.setitem(sys.modules,loader_name,load_builtins)
            moc_import_lib.setattr("importlib.util.spec_from_loader",patch_importlib_util_spec_from_tests_loader)
            moc_import_lib.setattr("hickle.lookup.spec_from_loader",patch_importlib_util_spec_from_tests_loader)
            py_obj_type,nopickleloader = loader.load_loader(py_object.__class__)
            assert py_obj_type is dict and nopickleloader == (load_builtins.create_package_test,b'dict',True)

            # simulate loading of package or local loader from hickle_loaders directory
            backup_load_builtins = sys.modules.pop('hickle.loaders.load_builtins',None)
            backup_py_obj_type = loader.types_dict.pop(dict,None)
            backup_loaded_loaders = lookup.LoaderManager.__loaded_loaders__.discard('hickle.loaders.load_builtins')
            moc_import_lib.setattr("importlib.util.find_spec",patch_importlib_util_find_spec_no_load_builtins)
            moc_import_lib.setattr("hickle.lookup.find_spec",patch_importlib_util_find_spec_no_load_builtins)
            py_obj_type,nopickleloader = loader.load_loader(py_object.__class__)
            assert py_obj_type is dict 
            assert nopickleloader == (sys.modules['hickle.loaders.load_builtins'].create_package_test,b'dict',True)
            ## back to start test successful fallback to legacy .pyc in case no source is available for package
            sys.modules.pop('hickle.loaders.load_builtins','None')
            loader.types_dict.pop(dict,None)
            lookup.LoaderManager.__loaded_loaders__.discard('hickle.loaders.load_builtins')
            pyc_path = load_builtins.__file__ + 'c'
            if not os.path.isfile(pyc_path):
                compileall.compile_file(load_builtins.__file__,legacy=True)
                assert os.path.isfile(pyc_path)
            base_dir,base_name = os.path.split(load_builtins.__file__)
            hidden_source = os.path.join(base_dir,'.{}h'.format(base_name))
            os.rename(load_builtins.__file__,hidden_source)
            py_obj_type,nopickleloader = loader.load_loader(py_object.__class__)
            assert py_obj_type is dict 
            assert nopickleloader == (sys.modules['hickle.loaders.load_builtins'].create_package_test,b'dict',True)
            #once again just checking that if no legacy .pyc next base is tried
            sys.modules.pop('hickle.loaders.load_builtins','None')
            loader.types_dict.pop(dict,None)
            lookup.LoaderManager.__loaded_loaders__.discard('hickle.loaders.load_builtins')
            os.remove(pyc_path)
            py_obj_type,nopickleloader = loader.load_loader(py_object.__class__)
            assert py_obj_type is dict
            assert nopickleloader == (lookup.create_pickled_dataset,b'pickle',True)
            os.rename(hidden_source,load_builtins.__file__)
            moc_import_lib.setattr("importlib.util.spec_from_loader",patch_importlib_util_spec_from_loader)
            moc_import_lib.setattr("hickle.lookup.spec_from_loader",patch_importlib_util_spec_from_loader)
            moc_import_lib.setattr("importlib.util.find_spec",patch_importlib_util_find_spec)
            moc_import_lib.setattr("hickle.lookup.find_spec",patch_importlib_util_find_spec)
            sys.modules['hickle.loaders.load_builtins'] = backup_load_builtins
            loader.types_dict[dict] = backup_py_obj_type
            # not added by missing legacy .pyc test readd manually here
            lookup.LoaderManager.__loaded_loaders__.add('hickle.loaders.load_builtins')
            lookup._custom_loader_enabled_builtins.pop(py_obj_type.__class__.__module__,None)
    
            # preload dataset only loader and check that it can be resolved directly
            loader_spec = loader_table[0]
            lookup.LoaderManager.register_class(*loader_spec)
            assert loader.load_loader((12).__class__) == (loader_spec[0],(*loader_spec[2:0:-1],loader_spec[5]))
    
            # try to find appropriate loader for dict object, a mock of this
            # should have already been imported above 
            assert loader.load_loader(py_object.__class__) == (dict,(load_builtins.create_package_test,b'dict',True))
    
            # remove loader again and undo redirection again. dict should now be
            # processed by create_pickled_dataset
            moc_import_lib.delitem(sys.modules,loader_name)
            del lookup.LoaderManager.__py_types__[None][dict]
            py_obj_type,nopickleloader = loader.load_loader(py_object.__class__)
            assert py_obj_type is dict and nopickleloader == (lookup.create_pickled_dataset,b'pickle',True)
            
            # check that load_loader prevents redefinition of loaders to be predefined by hickle core
            with pytest.raises(
                RuntimeError,
                match = r"objects\s+defined\s+by\s+hickle\s+core\s+must\s+be"
                        r"\s+registered\s+before\s+first\s+dump\s+or\s+load"
            ):
                py_obj_type,nopickleloader = loader.load_loader(ToBeInLoadersOrNotToBe)
            moc_import_lib.setattr(ToBeInLoadersOrNotToBe,'__module__','hickle.loaders')
    
            # check that load_loaders issues drop warning upon loader definitions for
            # dummy objects defined within hickle package but outside loaders modules
            with pytest.warns(
                RuntimeWarning,
                match = r"ignoring\s+'.+'\s+dummy\s+type\s+not\s+defined\s+by\s+loader\s+module"
            ):
                py_obj_type,nopickleloader = loader.load_loader(ToBeInLoadersOrNotToBe)
                assert py_obj_type is ToBeInLoadersOrNotToBe
                assert nopickleloader == (lookup.create_pickled_dataset,b'pickle',True)
    
            # check that loader definitions for dummy objects defined by loaders work as expected
            # by loader module 
            moc_import_lib.setattr(ToBeInLoadersOrNotToBe,'__module__',loader_name)
            py_obj_type,(create_dataset,base_type,memoise) = loader.load_loader(ToBeInLoadersOrNotToBe)
            assert py_obj_type is ToBeInLoadersOrNotToBe and base_type == b'NotHicklable'
            assert create_dataset is not_dumpable
            assert memoise == False
    
            # remove loader_name from list of loaded loaders and check that loader is loaded anew
            # and that values returned for dict object correspond to loader 
            # provided by freshly loaded loader module
            lookup.LoaderManager.__loaded_loaders__.remove(loader_name)
            py_obj_type,(create_dataset,base_type,memoise) = loader.load_loader(py_object.__class__)
            load_builtins_moc = sys.modules.get(loader_name,None)
            assert load_builtins_moc is not None
            loader_spec = load_builtins_moc.class_register[0]
            assert py_obj_type is dict and create_dataset is loader_spec[2]
            assert base_type is loader_spec[1]
            assert memoise == True
            # check that package path is properly resolved if package module
            # for which to find loader for is not found on sys. modules or 
            # its __spec__ attribute is set to None typically on __main__ or
            # builtins and other c modules
            lookup.LoaderManager.__loaded_loaders__.remove(loader_name)
            backup_module = ClassToDump.__module__
            moc_import_lib.setattr(ClassToDump,'__module__',re.sub(r'^\s*hickle\.','',ClassToDump.__module__))
            py_obj_type,(create_dataset,base_type,memoise) = loader.load_loader(ClassToDump)
            assert py_obj_type is ClassToDump
            assert create_dataset is lookup.create_pickled_dataset
            assert base_type == b'pickle' and memoise == True
            ClassToDump.__module__ = backup_module
            moc_import_lib.setattr("hickle.lookup.find_spec",patch_hide_collections_loader)
            py_obj_type,(create_dataset,base_type,memoise) = loader.load_loader(collections.OrderedDict)
            moc_import_lib.setattr("hickle.lookup.find_spec",patch_importlib_util_find_spec)
            assert py_obj_type is collections.OrderedDict
            assert create_dataset is sys.modules[loader_name].create_package_test
            assert base_type == b'dict' and memoise == True
            

def test_type_legacy_mro():
    """
    tests type_legacy_mro function which is used in replacement
    for native type.mro function when loading 4.0.0 and 4.0.1 files
    it handles cases where type objects passed to load_loader are
    functions not classes
    """

    # check that for class object type_legacy_mro function returns
    # the mro list provided by type.mro unchanged
    assert lookup.type_legacy_mro(SimpleClass) == type.mro(SimpleClass)

    # check that in case function is passed as type object a tuple with
    # function as single element is returned
    assert lookup.type_legacy_mro(function_to_dump) == (function_to_dump,)


def test_create_pickled_dataset(h5_data,compression_kwargs):
    """
    tests create_pickled_dataset, load_pickled_data function and PickledContainer 
    """
    
    # check if create_pickled_dataset issues SerializedWarning for objects which
    # either do not support copy protocol
    py_object = ClassToDump('hello',1)
    pickled_py_object = pickle.dumps(py_object)
    data_set_name = "greetings"
    with pytest.warns(lookup.SerializedWarning,match = r".*type\s+not\s+understood,\s+data\s+is\s+serialized:.*") as warner:
        h5_node,subitems = lookup.create_pickled_dataset(py_object, h5_data,data_set_name,**compression_kwargs)
        assert isinstance(h5_node,h5py.Dataset) and not subitems and iter(subitems)
        assert bytes(h5_node[()]) == pickled_py_object and h5_node.name.rsplit('/',1)[-1] == data_set_name
        assert lookup.load_pickled_data(h5_node,b'pickle',object) == py_object
        backup_class_to_dump = globals()['ClassToDump']
        backup_class_to_dump = globals().pop('ClassToDump',None)
        recovered = lookup.load_pickled_data(h5_node,b'pickle',object)
        assert isinstance(recovered,lookup.RecoveredDataset)
        assert bytes(recovered) == pickled_py_object
        globals()['ClassToDump'] = backup_class_to_dump
    
    
def test__DictItemContainer():
    """
    tests _DictItemContainer class which represent dict_item group 
    used by version 4.0.0 files to represent values of dictionary key
    """
    container = lookup._DictItemContainer({},b'dict_item',lookup._DictItem)
    my_bike_lock = (1,2,3,4)
    container.append('my_bike_lock',my_bike_lock,{})
    assert container.convert() is my_bike_lock

    
#@pytest.mark.no_compression
def test__moc_numpy_array_object_lambda():
    """
    test the _moc_numpy_array_object_lambda function
    which mimics the effect of lambda function created
    py pickle when expanding pickle `'type'` string set
    for numpy arrays containing a single object not expandable
    into a list. Mocking is necessary from Python 3.8.X on
    as it seems in Python 3.8 and onward trying to pickle
    a lambda now causes a TypeError whilst it seems to be silently
    accepted in Python < 3.8
    """
    data = ['hello','world']
    assert lookup._moc_numpy_array_object_lambda(data) == data[0]
    
#@pytest.mark.no_compression
def test_fix_lambda_obj_type():
    """
    test _moc_numpy_array_object_lambda function it self. When invoked
    it should return the first element of the passed list
    """
    assert lookup.fix_lambda_obj_type(None) is object
    picklestring = pickle.dumps(SimpleClass)
    assert lookup.fix_lambda_obj_type(picklestring) is SimpleClass
    with pytest.warns(lookup.MockedLambdaWarning):
        assert lookup.fix_lambda_obj_type('') is lookup._moc_numpy_array_object_lambda

def test_ReferenceManager_get_root(h5_data):
    """
    tests the static ReferenceManager._get_root method
    """

    # create an artificial 'hickle_types_table' with some entries
    # and link their h5py.Reference objects to the 'type' attributes 
    # of some data such that ReferenceManager._get_root can resolve
    # h5_data root_group independent which node it was passed
    
    root_group = h5_data['/root_group']
    data_group = root_group.create_group('data')
    content = data_group.create_dataset('mydata',data=12)
    type_table = root_group.create_group('hickle_types_table')
    int_pickle_string = bytearray(pickle.dumps(int))
    int_np_entry = np.array(int_pickle_string,copy=False)
    int_np_entry.dtype = 'S1'
    int_entry = type_table.create_dataset(str(len(type_table)),data = int_np_entry,shape =(1,int_np_entry.size))
    int_base_type = b'int'
    int_base_type = type_table.create_dataset(int_base_type,shape=None,dtype="S1")
    int_entry.attrs['base_type'] = int_base_type.ref
    content.attrs['type'] = int_entry.ref
    # try to resolve root_group from various kinds of nodes including 
    # root_group it self.
    assert lookup.ReferenceManager.get_root(content).id == root_group.id
    assert lookup.ReferenceManager.get_root(root_group).id == root_group.id
    assert lookup.ReferenceManager.get_root(data_group).id == data_group.id

    # check fallbacks to passe in group or file in case resolution via
    # 'type' attribute reference fails
    list_group = data_group.create_group('somelist')
    some_list_item = list_group.create_dataset('0',data=13)
    assert lookup.ReferenceManager.get_root(some_list_item).id == some_list_item.file.id
    assert lookup.ReferenceManager.get_root(list_group).id == list_group.id

    # test indirect resolution through 'type' reference of parent group
    # which should have an already properly assigned 'type' attribute
    # unless reading hickle 4.0.X file or referred to 'hickle_types_table' entry
    # is missing. In both cases file shall be returned as fallback
    list_pickle_string = bytearray(pickle.dumps(list))
    list_np_entry = np.array(list_pickle_string,copy = False)
    list_np_entry.dtype = 'S1'
    list_entry = type_table.create_dataset(str(len(type_table)),data = list_np_entry,shape=(1,list_np_entry.size))
    list_base_type = b'list'
    list_base_type = type_table.create_dataset(list_base_type,shape=None,dtype="S1")
    list_entry.attrs['base_type'] = list_base_type.ref
    list_group.attrs['type'] = list_pickle_string
    assert lookup.ReferenceManager.get_root(some_list_item).id == root_group.file.id
    list_group.attrs['type'] = list_entry.ref
    assert lookup.ReferenceManager.get_root(some_list_item).id == root_group.id
    for_later_use = list_entry.ref
    list_entry = None
    del type_table[str(len(type_table)-2)]
    assert lookup.ReferenceManager.get_root(some_list_item).id == root_group.file.id
    assert lookup.ReferenceManager.get_root(some_list_item).id == root_group.file.id
    
        

class not_a_surviver():
    """does not survive pickle.dumps"""

def test_ReferenceManager(h5_data):
    """
    test for creation of ReferenceManager object (__init__)
    to be run before testing ReferenceManager.create_manager
    """
    
    reference_manager = lookup.ReferenceManager(h5_data)
    type_table = h5_data['hickle_types_table']
    assert isinstance(type_table,h5py.Group)
    reference_manager = lookup.ReferenceManager(h5_data)
    assert reference_manager._py_obj_type_table.id == type_table.id
    false_root = h5_data.file.create_group('false_root')
    false_root.create_dataset('hickle_types_table',data=12)
    with pytest.raises(lookup.ReferenceError):
        reference_manager = lookup.ReferenceManager(false_root)
    int_pickle_string = bytearray(pickle.dumps(int))
    int_np_entry = np.array(int_pickle_string,copy=False)
    int_np_entry.dtype = 'S1'
    int_entry = type_table.create_dataset(str(len(type_table)),data = int_np_entry,shape =(1,int_np_entry.size))
    int_base_type = b'int'
    int_base_type = type_table.create_dataset(int_base_type,shape=None,dtype="S1")
    int_entry.attrs['base_type'] = int_base_type.ref
    list_pickle_string = bytearray(pickle.dumps(list))
    list_np_entry = np.array(list_pickle_string,copy = False)
    list_np_entry.dtype = 'S1'
    list_entry = type_table.create_dataset(str(len(type_table)),data = list_np_entry,shape=(1,list_np_entry.size))
    list_base_type = b'list'
    list_base_type = type_table.create_dataset(list_base_type,shape=None,dtype="S1")
    list_entry.attrs['base_type'] = list_base_type.ref

    missing_pickle_string = bytearray(pickle.dumps(not_a_surviver))
    missing_np_entry = np.array(missing_pickle_string,copy = False)
    missing_np_entry.dtype = 'S1'
    missing_entry = type_table.create_dataset(str(len(type_table)),data = missing_np_entry,shape=(1,missing_np_entry.size))
    missing_base_type = b'lost'
    missing_base_type = type_table.create_dataset(missing_base_type,shape=None,dtype="S1")
    missing_entry.attrs['base_type'] = missing_base_type.ref
    hide_not_a_surviver = globals().pop('not_a_surviver',None)
    reference_manager = lookup.ReferenceManager(h5_data)
    globals()['not_a_surviver'] = hide_not_a_surviver
    assert reference_manager._py_obj_type_link[id(int)] == int_entry
    assert reference_manager._py_obj_type_link[int_entry.id] == (int,b'int')
    assert reference_manager._base_type_link[b'int'] == int_base_type
    assert reference_manager._base_type_link[int_base_type.id] == b'int'
    assert reference_manager._py_obj_type_link[id(list)] == list_entry
    assert reference_manager._py_obj_type_link[list_entry.id] == (list,b'list')
    assert reference_manager._base_type_link[b'list'] == list_base_type
    assert reference_manager._base_type_link[list_base_type.id] == b'list'
    assert reference_manager._base_type_link[b'lost'] == missing_base_type
    assert reference_manager._base_type_link[missing_base_type.id] == b'lost'
    assert reference_manager._py_obj_type_link[missing_entry.id] == (lookup.AttemptRecoverCustom,'!recover!',b'lost')
    backup_attr = list_entry.attrs['base_type']
    list_entry.attrs.pop('base_type',None)
    with pytest.raises(lookup.ReferenceError):
        reference_manager = lookup.ReferenceManager(h5_data)
    list_entry.attrs['base_type']=b'list'
    with pytest.raises(lookup.ReferenceError):
        reference_manager = lookup.ReferenceManager(h5_data)
    stale_ref_entry = type_table.create_dataset("stale",shape=None,dtype = 'S1')
    list_entry.attrs['base_type']=stale_ref_entry.ref
    type_table.pop("stale",None)
    stale_ref_entry = None
    with pytest.raises(lookup.ReferenceError):
        reference_manager = lookup.ReferenceManager(h5_data)
    list_entry.attrs['base_type']=backup_attr
    
    old_hickle_file_root = h5_data.file.create_group('old_root')
    h5_data.file.flush()
    base_name,ext = h5_data.file.filename.rsplit('.',1)
    file_name = "{}_ro.{}".format(base_name,ext)
    shutil.copyfile(h5_data.file.filename,file_name)
    data_name = h5_data.name
    read_only_handle = h5py.File(file_name,'r')
    h5_read_data = read_only_handle[data_name]
    h5_read_old = read_only_handle['old_root']
    reference_manager = lookup.ReferenceManager(h5_read_old)
    assert isinstance(reference_manager._overlay,weakref.finalize)
    overlay_file = reference_manager._py_obj_type_table.file
    assert overlay_file.mode == 'r+' and overlay_file.driver == 'core'
    assert overlay_file.id != read_only_handle.id
    reference_manager = lookup.ReferenceManager(h5_read_data)
    
    
    read_only_handle.close()
    class SubReferenceManager(lookup.ReferenceManager):
        __managers__ = ()
    assert SubReferenceManager.__managers__ is lookup.ReferenceManager.__managers__

    with pytest.raises(TypeError):
        invalid_instance = lookup.BaseManager()

    class OtherManager(lookup.BaseManager):
        pass

    with pytest.raises(NotImplementedError):
        with OtherManager() as invalid_manager:
            pass

    

    
def test_ReferenceManager_drop_manager(h5_data):
    """
    test static ReferenceManager._drop_table method
    """
    reference_manager = lookup.ReferenceManager(h5_data)
    lookup.ReferenceManager.__managers__[h5_data.file.id] = (reference_manager,h5_data)
    some_other_file = h5py.File('someother.hdf5','w')
    some_other_root = some_other_file.create_group('root')
    lookup.ReferenceManager._drop_manager(some_other_root.file.id)
    lookup.ReferenceManager.__managers__[some_other_file.file.id] = (lookup.ReferenceManager(some_other_root),some_other_root)
    assert lookup.ReferenceManager.__managers__.get(h5_data.file.id,None) == (reference_manager,h5_data)
    lookup.ReferenceManager._drop_manager(h5_data.file.id)
    assert lookup.ReferenceManager.__managers__.get(h5_data.file.id,None) is None
    lookup.ReferenceManager._drop_manager(some_other_root.file.id)
    assert not lookup.ReferenceManager.__managers__
    some_other_file.close()
    
def test_ReferenceManager_create_manager(h5_data):
    """
    test public static ReferenceManager.create_manager function
    """
    second_tree = h5_data.file.create_group('seondary_root')
    h5_data_table = lookup.ReferenceManager.create_manager(h5_data)
    assert lookup.ReferenceManager.__managers__[h5_data.file.id][0] is h5_data_table
    with pytest.raises(lookup.LookupError):
        second_table = lookup.ReferenceManager.create_manager(second_tree)
    lookup.ReferenceManager._drop_manager(h5_data.file.id)

def test_ReferenceManager_context(h5_data):
    """
    test use of ReferenceManager as context manager
    """
    with lookup.ReferenceManager.create_manager(h5_data) as memo:
        assert lookup.ReferenceManager.__managers__[h5_data.file.id][0] is memo
    assert memo._py_obj_type_table is None
    with pytest.raises(RuntimeError):
        with memo as memo2:
            pass
    memo.__exit__(None,None,None)
    old_hickle_file_root = h5_data.file.create_group('old_root')
    h5_data.file.flush()
    base_name,ext = h5_data.file.filename.rsplit('.',1)
    file_name = "{}_ro.{}".format(base_name,ext)
    shutil.copyfile(h5_data.file.filename,file_name)
    data_name = old_hickle_file_root.name
    read_only_handle = h5py.File(file_name,'r')
    h5_read_data = read_only_handle[data_name]
    with lookup.ReferenceManager.create_manager(h5_read_data) as memo:
        assert isinstance(memo._overlay,weakref.finalize)
    assert memo._overlay is None
    read_only_handle.close()
        
def test_ReferenceManager_store_type(h5_data,compression_kwargs):
    """
    test ReferenceManager.store_type method which sets 'type' attribute
    reference to appropriate py_obj_type entry within 'hickle_types_table'
    """
    h_node = h5_data.create_group('some_list')
    with lookup.ReferenceManager.create_manager(h5_data) as memo:
        memo.store_type(h_node,object,None,**compression_kwargs)
        assert len(memo._py_obj_type_table) == 0 and not memo._py_obj_type_link and not memo._base_type_link
        with pytest.raises(lookup.LookupError):
            memo.store_type(h_node,list,None,**compression_kwargs)
        with pytest.raises(ValueError):
            memo.store_type(h_node,list,b'',**compression_kwargs)
        memo.store_type(h_node,list,b'list',**compression_kwargs)
        assert isinstance(h_node.attrs['type'],h5py.Reference)
        type_table_entry = h5_data.file[h_node.attrs['type']]
        assert pickle.loads(type_table_entry[()]) is list
        assert isinstance(type_table_entry.attrs['base_type'],h5py.Reference)
        assert h5_data.file[type_table_entry.attrs['base_type']].name.rsplit('/',1)[-1].encode('ascii') == b'list'
    
@pytest.mark.no_compression
def test_ReferenceManager_get_manager(h5_data):
    h_node = h5_data.create_group('some_list')
    item_data = np.array(memoryview(b'hallo welt lore grueszet dich ipsum aus der lore von ipsum gelort in ipsum'),copy=False)
    item_data.dtype = 'S1'
    h_item = h_node.create_dataset('0',data=item_data,shape=(1,item_data.size))
    with lookup.ReferenceManager.create_manager(h5_data) as memo:
        memo.store_type(h_node,list,b'list')
        memo.store_type(h_item,bytes,b'bytes')
        assert lookup.ReferenceManager.get_manager(h_item) == memo

        backup_manager = lookup.ReferenceManager.__managers__.pop(h5_data.file.id,None)
        assert backup_manager is not None
        with pytest.raises(lookup.ReferenceError):
            manager = lookup.ReferenceManager.get_manager(h_item)
        lookup.ReferenceManager.__managers__[h5_data.file.id] = backup_manager
    with pytest.raises(lookup.ReferenceError):
        manager = lookup.ReferenceManager.get_manager(h_item)

@pytest.mark.no_compression
def test_ReferenceManager_resolve_type(h5_data):
    """
    test ReferenceManager.reslove_type method which tries to resolve
    content of 'type' attribute of passed in node and return appropriate pair of
    py_obj_type,base_type and a boolean flag indicating whether node represents
    a h5py.Group or h5py.Reference both of which are to be handled by PyContainer
    objects.
    """
    invalid_pickle_and_ref = h5_data.create_group('invalid_pickle_and_ref')
    pickled_data = h5_data.create_dataset('pickled_data',data = bytearray())
    shared_ref = h5_data.create_dataset('shared_ref',data = pickled_data.ref,dtype = h5py.ref_dtype)
    old_style_typed = h5_data.create_dataset('old_style_typed',data = 12)
    old_style_typed.attrs['type'] = np.array(pickle.dumps(int))
    old_style_typed.attrs['base_type'] = b'int'
    broken_old_style = h5_data.create_dataset('broken_old_style',data = 12)
    broken_old_style.attrs['type'] = 12
    broken_old_style.attrs['base_type'] = b'int'
    new_style_typed = h5_data.create_dataset('new_style_typed',data = 12)
    stale_new_style = h5_data.create_dataset('stale_new_style',data = 12)
    new_style_typed_no_link = h5_data.create_dataset('new_style_typed_no_link',data = 12.5)
    has_not_recoverable_type = h5_data.create_dataset('no_recoverable_type',data = 42.56)
    with lookup.ReferenceManager.create_manager(h5_data) as memo:
        with pytest.raises(lookup.ReferenceError):
            memo.resolve_type(invalid_pickle_and_ref)
        assert memo.resolve_type(pickled_data) == (object,b'pickle',False)
        assert memo.resolve_type(shared_ref) == (lookup.NodeReference,b'!node-reference!',True)
        assert memo.resolve_type(old_style_typed) in ((int,b'int',False),(int,'int',False))
        with pytest.raises(lookup.ReferenceError):
            info = memo.resolve_type(broken_old_style)
        memo.store_type(new_style_typed,int,b'int')
        entry_id = len(memo._py_obj_type_table)
        memo.store_type(stale_new_style,list,b'list')
        assert pickle.loads(memo._py_obj_type_table[str(entry_id)][()]) is list
        stale_list_base = memo._py_obj_type_table['list'].ref
        # remove py_obj_type entry and base_type_entry for list entry
        # while h5py 2 raises a value error if not active link exists for a
        # dataset h5py 3 returns an anonymous group when resolving a stale 
        # reference to it if any body still holds a strong reference to its
        # h5py.Dataset or h5py.Group object. Therefore drop all references to
        # the removed entries to simulate that somebody has removed them from
        # a hickle file before it was passed to hickle.load for restoring its
        # content.
        memo._py_obj_type_link.pop(memo._py_obj_type_table[str(entry_id)].id,None)
        memo._py_obj_type_link.pop(id(list),None)
        memo._base_type_link.pop(memo._py_obj_type_table['list'].id,None)
        memo._base_type_link.pop(b'list',None)
        del memo._py_obj_type_table[str(entry_id)]
        del memo._py_obj_type_table['list']
        memo._py_obj_type_table.file.flush()
        with pytest.raises(lookup.ReferenceError):
            memo.resolve_type(stale_new_style)
        entry_id = len(memo._py_obj_type_table)
        memo.store_type(new_style_typed_no_link,float,b'float')
        float_entry = memo._py_obj_type_table[str(entry_id)]
        assert pickle.loads(float_entry[()]) is float
        float_base = float_entry.attrs['base_type']
        # remove float entry and clear all references to it see above 
        del memo._py_obj_type_link[float_entry.id]
        del memo._py_obj_type_link[id(float)]
        del float_entry.attrs['base_type']
        memo._py_obj_type_table.file.flush()
        assert memo.resolve_type(new_style_typed_no_link) in ((float,b'pickle',False),(float,'pickle',False))
        del memo._py_obj_type_link[float_entry.id]
        del memo._py_obj_type_link[id(float)]
        # create stale reference to not existing base_type entry
        memo._py_obj_type_table.create_dataset('list',shape=None,dtype='S1')
        float_entry.attrs['base_type'] = memo._py_obj_type_table['list'].ref
        memo._py_obj_type_table.pop('list',None)
        memo._py_obj_type_table.file.flush()
        with pytest.raises(lookup.ReferenceError):
            info = memo.resolve_type(new_style_typed_no_link)
        memo._py_obj_type_link.pop(float_entry.id,None)
        memo._py_obj_type_link.pop(id(float),None)
        del memo._base_type_link[memo._py_obj_type_table[float_base].id]
        del memo._base_type_link[b'float']
        float_entry.attrs['base_type'] = float_base
        memo._py_obj_type_table.file.flush()
        assert memo.resolve_type(new_style_typed_no_link) in ((float,b'float',False),(float,'float',False))
        
        assert memo.resolve_type(new_style_typed_no_link) 
        memo.store_type(has_not_recoverable_type,not_a_surviver,b'lost')
        del memo._py_obj_type_link[memo._py_obj_type_link[id(not_a_surviver)].id]
        del memo._py_obj_type_link[id(not_a_surviver)]
        hide_not_a_surviver = globals().pop('not_a_surviver',None)
        assert memo.resolve_type(has_not_recoverable_type) == (lookup.AttemptRecoverCustom,b'!recover!',False)
        assert memo.resolve_type(has_not_recoverable_type,base_type_type=2) == (lookup.AttemptRecoverCustom,b'lost',False)
        globals()['not_a_surviver'] = hide_not_a_surviver
        has_not_recoverable_type.attrs['type'] = np.array(pickle.dumps(not_a_surviver))
        has_not_recoverable_type.attrs['base_type'] = b'lost'
        hide_not_a_surviver = globals().pop('not_a_surviver',None)
        assert memo.resolve_type(has_not_recoverable_type) == (lookup.AttemptRecoverCustom,b'!recover!',False)
        assert memo.resolve_type(has_not_recoverable_type,base_type_type=2) in ((lookup.AttemptRecoverCustom,b'lost',False),(lookup.AttemptRecoverCustom,'lost',False))
        globals()['not_a_surviver'] = hide_not_a_surviver
    
    
def test_ExpandReferenceContainer(h5_data):
    """
    test ExpandReferenceContainer which resolves object link stored as h5py.Reference
    type dataset
    """
    expected_data = np.random.randint(-13,13,12)
    referred_data = h5_data.create_dataset('referred_data',data = expected_data)
    referring_node = h5_data.create_dataset('referring_node',data = referred_data.ref,dtype = h5py.ref_dtype)
    h5_data.file.flush()
    sub_container = lookup.ExpandReferenceContainer(referring_node.attrs,b'!node-reference!',lookup.NodeReference)
    content = None
    for name,subitem in sub_container.filter(referring_node):
        assert name == 'referred_data' and subitem.id == referred_data.id
        content = np.array(subitem[()])
        sub_container.append(name,content,subitem.attrs)
    assert np.all(sub_container.convert()==expected_data)
    referring_node = h5_data.create_dataset('stale_reference',shape=(),dtype=h5py.ref_dtype)
    sub_container = lookup.ExpandReferenceContainer(referring_node.attrs,b'!node-reference!',lookup.NodeReference)
    with pytest.raises(lookup.ReferenceError):
        for name,subitem in sub_container.filter(referring_node):
            content = np.array(subitem[()])
            sub_container.append(name,content,subitem.attrs)

@pytest.mark.no_compression
def test_recover_custom_data(h5_data):
    array_to_recover = np.random.random_sample([4,2])
    with lookup.ReferenceManager.create_manager(h5_data) as memo:
        dataset_to_recover = h5_data.create_dataset('to_recover',data=array_to_recover)
        dataset_to_recover.attrs['world'] = 2
        memo.store_type(dataset_to_recover,ClassToDump,b'myclass')
        group_to_recover = h5_data.create_group('need_recover')
        memo.store_type(group_to_recover,ClassToDump,b'myclass')
        backup_class_to_dump = globals().pop('ClassToDump',None)
        memo._py_obj_type_link.pop(id('ClassToDump'),None)
        memo._base_type_link.pop(b'myclass')
        type_entry = memo._py_obj_type_table[dataset_to_recover.attrs['type']]
        memo._py_obj_type_link.pop(type_entry.id,None)
        py_obj_type,base_type,is_group = memo.resolve_type(dataset_to_recover)
        assert issubclass(py_obj_type,lookup.AttemptRecoverCustom) and base_type == b'!recover!'
        with pytest.warns(lookup.DataRecoveredWarning):
            recovered = lookup.recover_custom_dataset(dataset_to_recover,base_type,py_obj_type) 
        assert recovered.dtype == array_to_recover.dtype and np.all(recovered == array_to_recover)
        assert recovered.attrs == {'base_type':b'myclass','world':2}
        assert not is_group
        type_entry = memo._py_obj_type_table[group_to_recover.attrs['type']]
        memo._py_obj_type_link.pop(type_entry.id,None)
        some_int=group_to_recover.create_dataset('some_int',data=42)
        some_float=group_to_recover.create_dataset('some_float',data=42.0)
        group_to_recover.attrs['so'] = 'long'
        group_to_recover.attrs['and'] = 'thanks'
        some_float.attrs['for'] = 'all'
        some_float.attrs['the'] = 'fish'
        py_obj_type,base_type,is_group = memo.resolve_type(group_to_recover)
        assert issubclass(py_obj_type,lookup.AttemptRecoverCustom) and base_type == b'!recover!'
        assert is_group
        recover_container = lookup.RecoverGroupContainer(group_to_recover.attrs,base_type,py_obj_type)
        with pytest.warns(lookup.DataRecoveredWarning):
            for name,item in recover_container.filter(group_to_recover):
                recover_container.append(name,item[()],item.attrs)
        recover_container.append('some_other',recovered,recovered.attrs)
        recovered_group = recover_container.convert()
        assert isinstance(recovered_group,dict)
        assert some_float[()] == recovered_group['some_float'][0] and some_float.attrs == recovered_group['some_float'][1]
        assert some_int[()] == recovered_group['some_int'][0] and some_int.attrs == recovered_group['some_int'][1]
        assert recovered_group['some_other'] is recovered
        assert recovered_group.attrs['base_type'] == memo.resolve_type(group_to_recover,base_type_type=2)[1]
        assert len(recovered_group.attrs) == 3
        assert recovered_group.attrs['so'] == 'long' and recovered_group.attrs['and'] == 'thanks'
        globals()['ClassToDump'] = backup_class_to_dump 

if __name__ == "__main__":
    from _pytest.monkeypatch import monkeypatch
    from _pytest.fixtures import FixtureRequest
    from hickle.tests.conftest import compression_kwargs

    for h5_root in h5_data(FixtureRequest(test_create_pickled_dataset)):
        test_AttemptRecoverCustom_classes(h5_data)
    for table in loader_table():
        test_LoaderManager_register_class(table)
    for table in loader_table():
        test_LoaderManager_register_class_exclude(table)
    for table,h5_root in (
        (tab,root)
        for tab in loader_table()
        for root in h5_data(FixtureRequest(test_LoaderManager))
    ):
        test_LoaderManager(table,h5_root)
    for h5_root in h5_data(FixtureRequest(test_LoaderManager_drop_manager)):
        test_LoaderManager_drop_manager(h5_root)
    for h5_root in h5_data(FixtureRequest(test_LoaderManager_create_manager)):
        test_LoaderManager_create_manager(h5_root)
    for h5_root in h5_data(FixtureRequest(test_LoaderManager_context)):
        test_LoaderManager_context(h5_root)
    for table,h5_root,monkey in (
        (tab,root,mpatch)
        for tab in loader_table()
        for root in h5_data(FixtureRequest(test_LoaderManager_load_loader))
        for mpatch in monkeypatch()
    ):
            test_LoaderManager_load_loader(table,h5_root,monkey)
    test_type_legacy_mro()
    for h5_root,keywords in (
        ( h5_data(request),compression_kwargs(request) )
        for request in (FixtureRequest(test_create_pickled_dataset),)
    ):
        test_create_pickled_dataset(h5_root,keywords)
    test__DictItemContainer()
    test__moc_numpy_array_object_lambda()
    test_fix_lambda_obj_type()
    test_fix_lambda_obj_type()
    for h5_root in h5_data(FixtureRequest(test_ReferenceManager_get_root)):
        test_ReferenceManager_get_root(h5_root)
    for h5_root in h5_data(FixtureRequest(test_ReferenceManager)):
        test_ReferenceManager(h5_root)
    for h5_root in h5_data(FixtureRequest(test_ReferenceManager_drop_manager)):
        test_ReferenceManager_drop_manager(h5_root)
    for h5_root in h5_data(FixtureRequest(test_ReferenceManager_create_manager)):
        test_ReferenceManager_create_manager(h5_root)
    for h5_root in h5_data(FixtureRequest(test_ReferenceManager_context)):
        test_ReferenceManager_context(h5_root)
    for h5_root in h5_data(FixtureRequest(test_ReferenceManager_get_manager)):
        test_ReferenceManager_get_manager(h5_root)
    for h5_root,compression_kwargs in (
            h5_data(FixtureRequest(test_ReferenceManager_store_type))
    ):
        test_ReferenceManager_store_type(h5_root,compression_kwargs)
    for h5_root in h5_data(FixtureRequest(test_ReferenceManager_resolve_type)):
        test_ReferenceManager_resolve_type(h5_root)
    for h5_root in h5_data(FixtureRequest(test_ExpandReferenceContainer)):
        test_ExpandReferenceContainer(h5_root)
    for h5_root in h5_data(FixtureRequest(test_ExpandReferenceContainer)):
        test_recover_custom_data(h5_data)


    
    

