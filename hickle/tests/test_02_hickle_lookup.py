#! /usr/bin/env python
# encoding: utf-8
"""
# test_hickle_lookup.py

Unit tests for hickle module -- lookup functions.

"""

# %% IMPORTS
import pytest
import sys
import types

# Package imports
import numpy as np
import h5py
import dill as pickle
from importlib.util import find_spec
from importlib import reload
from py.path import local

# hickle imports
from hickle.helpers import PyContainer,not_dumpable
import hickle.lookup as lookup

# Set current working directory to the temporary directory
local.get_temproot().chdir()

# %% DATA DEFINITIONS

dummy_data = (1,2,3)


# %% FIXTURES

@pytest.fixture
def h5_data(request):
    """
    create dummy hdf5 test data file for testing PyContainer and H5NodeFilterProxy
    uses name of executed test as part of filename
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
    4: hickle loader moudle trying to overload hickle core loader
    3: loader defined by hickle core
    
    """

    # clear loaded_loaders, types_dict, hkl_types_dict and hkl_contianer_dict
    # to ensure no loader preset by hickle core or hickle loader module
    # intervenes with test
    global lookup
    lookup.loaded_loaders.clear()
    lookup.types_dict.clear()
    lookup.hkl_types_dict.clear()
    lookup.hkl_container_dict.clear()

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
        (int,b'int',create_test_dataset,load_test_dataset,None),
        (list,b'list',create_test_dataset,None,TestContainer),
        (tuple,b'tuple',None,load_test_dataset,TestContainer),
        (lookup._DictItem,b'dict_item',None,None,NotHicklePackage),
        (lookup._DictItem,b'pickle',None,None,HickleLoadersModule),
        (lookup._DictItem,b'dict_item',lookup.register_class,None,IsHickleCore)
    ]

    # cleanup and reload hickle.lookup module to reset it to its initial state
    # in case hickle.hickle has already been preloaded by pytest also reload it
    # to ensure no sideffectes occur during later tests
    lookup.loaded_loaders.clear()
    lookup.types_dict.clear()
    lookup.hkl_types_dict.clear()
    lookup.hkl_container_dict.clear()
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
            

class MetaClassToDump(type):
    """
    Metaclass for ClassToDump allowing to controll which
    unbound class methods and magic methods are visible to
    create_pickled_dataset method and which not at the
    class level
    """

    # any function listed therein is not defined on class
    # when called the next time (single shot)
    hide_special = set()

    def __getattribute__(self,name):
        if name in MetaClassToDump.hide_special:
            MetaClassToDump.hide_special.remove(name)
            raise AttributeError("")
        return super(MetaClassToDump,self).__getattribute__(name)
    

class ClassToDump(metaclass=MetaClassToDump):
    """
    Primary class used to test create_pickled_dataset function
    """
    def __init__(self,hallo,welt,with_default=1):
        self._data = hallo,welt,with_default

    def dump_boundmethod(self):
        """
        dummy instance method used to check if instance methods are
        either rejected or allways stored as pickle string
        """
        pass

    @staticmethod
    def dump_staticmethod():
        """
        dummy static method used to check if static methods are allways
        stored as pickle string
        """
        pass

    @classmethod
    def dump_classmethod(cls):
        """
        dummy class method used to check if class methods are allways
        stored as pickle string
        """
        pass

    def __eq__(self,other):
        return other.__class__ is self.__class__ and self._data == other._data

    def __ne__(self,other):
        return self != other

    def __getattribute__(self,name):
        # ensure that methods which are hidden by metaclass are also not
        # accessible from class instance
        if name in MetaClassToDump.hide_special:
            raise AttributeError("")
        return super(ClassToDump,self).__getattribute__(name)

    def __getstate__(self):
        # returns the state of this class when asked by copy protocol handler
        return self.__dict__

    def __setstate__(self,state):
        
        # set the state from the passed state description
        self.__dict__.update(state)

    # controls whether the setstate method is reported as
    # sixth element of tuple returned by __reduce_ex__ or
    # __reduce__ function or not
    extern_setstate = False

    def __reduce_ex__(self,proto = pickle.DEFAULT_PROTOCOL):
        state = super(ClassToDump,self).__reduce_ex__(proto)
        if len(state) > 5 or not ClassToDump.extern_setstate:
            return state
        return (*state,*( (None,) * ( 5 - len(state)) ),ClassToDump.__setstate__)

    def __reduce__(self):
        state = super(ClassToDump,self).__reduce__()
        if len(state) > 5 or not ClassToDump.extern_setstate:
            return state
        return (*state,*( (None,) * ( 5 - len(state)) ),ClassToDump.__setstate__)

class SimpleClass():
    """
    simple classe used to check that instance __dict__ is properly dumped and
    restored by create_pickled_dataset and PickledContainer
    """
    def __init__(self):
        self.someattr = "im some attr"
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
    non class function to be dumpled and restored through
    create_pickled_dataset and load_pickled_data
    """
    return hallo,welt,with_default

def test_register_class(loader_table):
    """
    tests the register_class method
    """

    # try to register dataset only loader specified by loader_table
    # and retrieve its contents from types_dict and hkl_types_dict
    loader_spec = loader_table[0]
    lookup.register_class(*loader_spec)
    assert lookup.types_dict[loader_spec[0]] == loader_spec[2:0:-1]
    assert lookup.hkl_types_dict[loader_spec[1]] == loader_spec[3]
    with pytest.raises(KeyError):
        lookup.hkl_container_dict[loader_spec[1]] is None

    # try to register PyContainer only loader specified by loader_table
    # and retrive its contents from types_dict and hkl_contianer_dict
    loader_spec = loader_table[1]
    lookup.register_class(*loader_spec)
    assert lookup.types_dict[loader_spec[0]] == loader_spec[2:0:-1]
    with pytest.raises(KeyError):
        lookup.hkl_types_dict[loader_spec[1]] is None
    assert lookup.hkl_container_dict[loader_spec[1]] == loader_spec[4]


    # try to register container without dump_function specified by
    # loader table and try to retrive load_function and PyContainer from
    # hkl_types_dict and hkl_container_dict
    loader_spec = loader_table[2]
    lookup.register_class(*loader_spec)
    with pytest.raises(KeyError):
        lookup.types_dict[loader_spec[0]][1] == loader_spec[1]
    assert lookup.hkl_types_dict[loader_spec[1]] == loader_spec[3]
    assert lookup.hkl_container_dict[loader_spec[1]] == loader_spec[4]

    # try to register loader shadowing loader preset by hickle core
    # defined by external loader module
    loader_spec = loader_table[3]
    with pytest.raises(TypeError,match = r"loader\s+for\s+'\w+'\s+type\s+managed\s+by\s+hickle\s+only"):
        lookup.register_class(*loader_spec)
    loader_spec = loader_table[4]

    # try to register loader shadowing loader preset by hickle core
    # defined by hickle loaders module
    with pytest.raises(TypeError,match = r"loader\s+for\s+'\w+'\s+type\s+managed\s+by\s+hickle\s+core\s+only"):
        lookup.register_class(*loader_spec)

    # simulate registering loader preset by hickle core
    loader_spec = loader_table[5]
    lookup.register_class(*loader_spec)

def test_register_class_exclude(loader_table):
    """
    test registr class exclude function
    """

    # try to disable loading of loader preset by hickle core
    base_type = loader_table[5][1]
    lookup.register_class(*loader_table[2])
    lookup.register_class(*loader_table[5])
    with pytest.raises(ValueError,match = r"excluding\s+'.+'\s+base_type\s+managed\s+by\s+hickle\s+core\s+not\s+possible"):
        lookup.register_class_exclude(base_type)

    # disable any of the other loaders
    base_type = loader_table[2][1]
    lookup.register_class_exclude(base_type)


def patch_importlib_util_find_spec(name,package=None):
    """
    function used to temporarily redirect search for laoders
    to hickle_loader directory in test directory for testing
    loading of new loaders
    """
    return find_spec("hickle.tests." + name.replace('.','_',1),package)

def patch_importlib_util_find_no_spec(name,package=None):
    """
    function used to simulate situation where no appropriate loader 
    could be found for object
    """
    return None


def test_load_loader(loader_table,monkeypatch):
    """
    test load_loader function
    """

    # some data to check loader for
    # assume loader should be load_builtins loader
    py_object = dict()
    loader_name = "hickle.loaders.load_builtins"
    with monkeypatch.context() as moc_import_lib:

        # hide loader from hickle.lookup.loaded_loaders and check that 
        # fallback loader for python object is returned
        moc_import_lib.setattr("importlib.util.find_spec",patch_importlib_util_find_no_spec)
        moc_import_lib.setattr("hickle.lookup.find_spec",patch_importlib_util_find_no_spec)
        moc_import_lib.delitem(sys.modules,"hickle.loaders.load_builtins",raising=False)
        py_obj_type,nopickleloader = lookup.load_loader(py_object.__class__)
        assert py_obj_type is dict and nopickleloader == (lookup.create_pickled_dataset,b'pickle')

        # redirect load_builtins loader to tests/hickle_loader path
        moc_import_lib.setattr("importlib.util.find_spec",patch_importlib_util_find_spec)
        moc_import_lib.setattr("hickle.lookup.find_spec",patch_importlib_util_find_spec)

        # preload dataset only loader and check that it can be resolved directly
        loader_spec = loader_table[0]
        lookup.register_class(*loader_spec)
        assert lookup.load_loader((12).__class__) == (loader_spec[0],loader_spec[2:0:-1])

        # try to find appropriate loader for dict object, a moc of this
        # loader should be provided by hickle/tests/hickle_loaders/load_builtins
        # module ensure that this module is the one found by load_loader function
        import hickle.tests.hickle_loaders.load_builtins as load_builtins
        moc_import_lib.setitem(sys.modules,loader_name,load_builtins)
        assert lookup.load_loader(py_object.__class__) == (dict,(load_builtins.create_package_test,b'dict'))

        # remove loader again and undo redirection again. dict should now be
        # processed by create_pickled_dataset
        moc_import_lib.delitem(sys.modules,loader_name)
        del lookup.types_dict[dict]
        py_obj_type,nopickleloader = lookup.load_loader(py_object.__class__)
        assert py_obj_type is dict and nopickleloader == (lookup.create_pickled_dataset,b'pickle')
        
        # check that load_loader prevenst redefinition of loaders to be predefined by hickle core
        with pytest.raises(
            RuntimeError,
            match = r"objects\s+defined\s+by\s+hickle\s+core\s+must\s+be"
                    r"\s+registerd\s+before\s+first\s+dump\s+or\s+load"
        ):
            py_obj_type,nopickleloader = lookup.load_loader(ToBeInLoadersOrNotToBe)
        monkeypatch.setattr(ToBeInLoadersOrNotToBe,'__module__','hickle.loaders')

        # check that load_loaders issues drop warning upon loader definitions for
        # dummy objects defined within hickle package but outsied loaders modules
        with pytest.warns(
            RuntimeWarning,
            match = r"ignoring\s+'.+'\s+dummy\s+type\s+not\s+defined\s+by\s+loader\s+module"
        ):
            py_obj_type,nopickleloader = lookup.load_loader(ToBeInLoadersOrNotToBe)
            assert py_obj_type is ToBeInLoadersOrNotToBe
            assert nopickleloader == (lookup.create_pickled_dataset,b'pickle')

        # check that loader definitions for dummy objets defined by loaders work as expected
        # by loader module 
        monkeypatch.setattr(ToBeInLoadersOrNotToBe,'__module__',loader_name)
        py_obj_type,(create_dataset,base_type) = lookup.load_loader(ToBeInLoadersOrNotToBe)
        assert py_obj_type is ToBeInLoadersOrNotToBe and base_type == b'NotHicklable'
        assert create_dataset is not_dumpable

        # remove loader_name from list of loaded loaders and check that loader is loaded anew
        # and that values returned for dict object correspond to loader 
        # provided by freshly loaded loader module
        lookup.loaded_loaders.remove(loader_name)
        py_obj_type,(create_dataset,base_type) = lookup.load_loader(py_object.__class__)
        load_builtins_moc = sys.modules.get(loader_name,None)
        assert load_builtins_moc is not None
        loader_spec = load_builtins_moc.class_register[0]
        assert py_obj_type is dict and create_dataset is loader_spec[2]
        assert base_type is loader_spec[1]

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


def test_create_pickled_dataset(h5_data):
    """
    tests create_pickled_dataset, load_pickled_data function and PickledContainer 
    """
    
    # check if create_pickled_dataset issues SerializedWarning for objects which
    # either do not support copy protocol
    py_object = ClassToDump('hello',1)
    data_set_name = "greetings"
    with pytest.warns(lookup.SerializedWarning,match = r".*type\s+not\s+understood,\s+data\s+is\s+serialized:.*") as warner:
        h5_node,subitems = lookup.create_pickled_dataset(py_object, h5_data,data_set_name)
        assert isinstance(h5_node,h5py.Dataset) and not subitems and iter(subitems)
        assert bytes(h5_node[()]) == pickle.dumps(py_object) and h5_node.name.split('/')[2] == data_set_name
        assert lookup.load_pickled_data(h5_node,b'pickle',object) == py_object
    
    
def test__DictItemContainer():
    """
    tests _DictItemContainer class which represent dict_item goup 
    used by version 4.0.0 files to represent values of dictionary key
    """
    container = lookup._DictItemContainer({},b'dict_item',lookup._DictItem)
    my_bike_lock = (1,2,3,4)
    container.append('my_bike_lock',my_bike_lock,{})
    assert container.convert() is my_bike_lock

    
def test__moc_numpy_array_object_lambda():
    """
    test the _moc_numpy_array_object_lambda function
    which mimicks the effect of lambda function created
    py pickle when expanding pickle `'type'` string set
    for numpy arrays containing a single object not expandable
    into a list. Mocking is necessary from Python 3.8.X on
    as it seems in Python 3.8 and onwards trying to pickle
    a lambda now causes a TypeError whilst it seems to be silently
    accepted in Python < 3.8
    """
    data = ['hello','world']
    assert lookup._moc_numpy_array_object_lambda(data) == data[0]
    
def test_fix_lambda_obj_type():
    """
    test _moc_numpy_array_object_lambda function it self. When invokded
    it should return the first element of the passed list
    """
    assert lookup.fix_lambda_obj_type(None) is object
    picklestring = pickle.dumps(SimpleClass)
    assert lookup.fix_lambda_obj_type(picklestring) is SimpleClass
    assert lookup.fix_lambda_obj_type('') is lookup._moc_numpy_array_object_lambda
    
# %% MAIN SCRIPT
if __name__ == "__main__":
    from _pytest.monkeypatch import monkeypatch
    from _pytest.fixtures import FixtureRequest
    for table in loader_table():
        test_register_class(table)
    for table in loader_table():
        test_register_class_exclude(table)
    for monkey in  monkeypatch():
        test_load_loader(table,monkey)
    test_type_legacy_mro()
    for h5_root in h5_data(FixtureRequest(test_create_pickled_dataset)):
        test_create_pickled_dataset(h5_root)
    test__DictItemContainer()
    test__moc_numpy_array_object_lambda()
    test_fix_lambda_obj_type()
    test_fix_lambda_obj_type()


    
    

