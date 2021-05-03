#! /usr/bin/env python
# encoding: utf-8
"""
# test_hickle.py

Unit test for hickle package.

"""


# %% IMPORTS

# Built-in imports
from collections import OrderedDict as odict
import os
import re
from pprint import pprint
import dill as pickle


# Package imports
import numpy as np
from py.path import local
import pytest

# hickle imports
from hickle import dump, hickle, load, lookup

# Set current working directory to the temporary directory
local.get_temproot().chdir()


# %% GLOBALS
NESTED_DICT = {
    "level1_1": {
        "level2_1": [1, 2, 3],
        "level2_2": [4, 5, 6]
    },
    "level1_2": {
        "level2_1": [1, 2, 3],
        "level2_2": [4, 5, 6]
    },
    "level1_3": {
        "level2_1": {
            "level3_1": [1, 2, 3],
            "level3_2": [4, 5, 6]
        },
        "level2_2": [4, 5, 6]
    }
}


# %% FIXTURES
    
@pytest.fixture
def test_file_name(request):
    """
    create test dependent filename path string
    """
    yield "{}.hkl".format(request.function.__name__)


# %% HELPER DEFINITIONS

# Define a test function that must be serialized and unpacked again
def func(a, b, c=0):
    """ just something to do """
    return(a, b, c)

# the following is required as package name of with_state is hickle
# and load_loader refuses load any loader module for classes defined inside
# hickle package exempt when defined within load_*.py loaders modules.
# That has to be done by hickle sub modules directly using register_class function
pickle_dumps = pickle.dumps
pickle_loads = pickle.loads

types_to_hide = set() 

def make_visible_to_dumps(obj,protocol=None,*,fix_imports=True):
    """
    simulate loader functions defined outside hickle package
    """
    if obj in types_to_hide:
        obj.__module__ = re.sub(r'^\s*(?!hickle\.)','hickle.',obj.__module__)
    elif obj.__class__ in types_to_hide:
        obj.__class__.__module__ = re.sub(r'^\s*(?!hickle\.)','hickle.',obj.__class__.__module__)
    return pickle_dumps(obj,protocol,fix_imports=fix_imports)

def hide_from_hickle(bytes_obj,*,fix_imports=True,encoding="ASCII",errors="strict"):
    """
    simulate loader function defined outside hickle package
    """
    obj = pickle_loads(bytes_obj,fix_imports = fix_imports, encoding = encoding, errors = errors)
    if obj in types_to_hide:
        obj.__module__ = re.sub(r'^\s*hickle\.','',obj.__module__)
    elif obj.__class__ in types_to_hide:
        obj.__class__.__module__ = re.sub(r'^\s*hickle\.','',obj.__class__.__module__)
    return obj

# Define a class that must always be pickled
class with_state(object):
    """
    A class that always must be handled by create_pickled_dataset
    """
    def __init__(self):
        self.a = 12
        self.b = {
            'love': np.ones([12, 7]),
            'hatred': np.zeros([4, 9])}

    def __getstate__(self):
        self.a *= 2
        return({
            'a': self.a,
            'b': self.b})

    def __setstate__(self, state):
        self.a = state['a']
        self.b = state['b']

    def __getitem__(self, index):
        if(index == 0):
            return(self.a)
        if(index < 2):
            return(self.b['hatred'])
        if(index > 2):
            raise ValueError("index unknown")
        return(self.b['love'])

types_to_hide.add(with_state)

# %% FUNCTION DEFINITIONS
def test_invalid_file():
    """ Test if trying to use a non-file object fails. """

    with pytest.raises(hickle.FileError):
        dump('test', ())


def test_state_obj(monkeypatch,test_file_name,compression_kwargs):
    """ Dumping and loading a class object with pickle states

    https://github.com/telegraphic/hickle/issues/125"""

    with monkeypatch.context() as monkey:
        monkey.setattr(with_state,'__module__',re.sub(r'^\s*hickle\.','',with_state.__module__))
        monkey.setattr(pickle,'dumps',make_visible_to_dumps)
        mode = 'w'
        obj = with_state()
        with pytest.warns(lookup.SerializedWarning):
            dump(obj, test_file_name, mode,**compression_kwargs)
        monkey.setattr(pickle,'loads',hide_from_hickle)
        obj_hkl = load(test_file_name)
        assert isinstance(obj,obj_hkl.__class__) or isinstance(obj_hkl,obj.__class__)
        assert np.allclose(obj[1], obj_hkl[1])


def test_local_func(test_file_name,compression_kwargs):
    """ Dumping and loading a local function

    https://github.com/telegraphic/hickle/issues/119"""

    mode =  'w'
    with pytest.warns(lookup.SerializedWarning):
        dump(func, test_file_name, mode,**compression_kwargs)
    func_hkl = load(test_file_name)
    assert isinstance(func,func_hkl.__class__) or isinstance(func_hkl,func.__class__)
    assert func(1, 2) == func_hkl(1, 2)


def test_non_empty_group(test_file_name,compression_kwargs):
    """ Test if attempting to dump to a group with data fails """

    hickle.dump(None, test_file_name,**compression_kwargs)
    with pytest.raises(ValueError):
        dump(None, test_file_name, 'r+',**compression_kwargs)


def test_string(test_file_name,compression_kwargs):
    """ Dumping and loading a string """
    mode = 'w'
    string_obj = "The quick brown fox jumps over the lazy dog"
    dump(string_obj, test_file_name, mode,**compression_kwargs)
    string_hkl = load(test_file_name)
    assert isinstance(string_hkl, str)
    assert string_obj == string_hkl


def test_65bit_int(test_file_name,compression_kwargs):
    """ Dumping and loading an integer with arbitrary precision

    https://github.com/telegraphic/hickle/issues/113"""
    i = 2**65-1
    dump(i, test_file_name,**compression_kwargs)
    i_hkl = load(test_file_name)
    assert i == i_hkl

    j = -2**63-1
    dump(j, test_file_name,**compression_kwargs)
    j_hkl = load(test_file_name)
    assert j == j_hkl

def test_list(test_file_name,compression_kwargs):
    """ Dumping and loading a list """
    filename, mode = 'test_list.h5', 'w'
    list_obj = [1, 2, 3, 4, 5]
    dump(list_obj, test_file_name, mode=mode,**compression_kwargs)
    list_hkl = load(test_file_name)
    try:
        assert isinstance(list_hkl, list)
        assert list_obj == list_hkl
        import h5py
        a = h5py.File(test_file_name, 'r')
        a.close()

    except AssertionError:
        print("ERR:", list_obj, list_hkl)
        import h5py

        raise


def test_set(test_file_name,compression_kwargs)    :
    """ Dumping and loading a list """
    mode = 'w'
    list_obj = set([1, 0, 3, 4.5, 11.2])
    dump(list_obj, test_file_name, mode,**compression_kwargs)
    list_hkl = load(test_file_name)
    try:
        assert isinstance(list_hkl, set)
        assert list_obj == list_hkl
    except AssertionError:
        print(type(list_obj))
        print(type(list_hkl))
        raise


def test_numpy(test_file_name,compression_kwargs):
    """ Dumping and loading numpy array """
    mode = 'w'
    dtypes = ['float32', 'float64', 'complex64', 'complex128']

    for dt in dtypes:
        array_obj = np.ones(8, dtype=dt)
        dump(array_obj, test_file_name, mode,**compression_kwargs)
        array_hkl = load(test_file_name)
    try:
        assert array_hkl.dtype == array_obj.dtype
        assert np.all((array_hkl, array_obj))
    except AssertionError:
        print(array_hkl)
        print(array_obj)
        raise


def test_masked(test_file_name,compression_kwargs):
    """ Test masked numpy array """
    mode = 'w'
    a = np.ma.array([1, 2, 3, 4], dtype='float32', mask=[0, 1, 0, 0])

    dump(a, test_file_name, mode,**compression_kwargs)
    a_hkl = load(test_file_name)

    try:
        assert a_hkl.dtype == a.dtype
        assert np.all((a_hkl, a))
    except AssertionError:
        print(a_hkl)
        print(a)
        raise


def test_object_numpy(test_file_name,compression_kwargs):
    """ Dumping and loading a NumPy array containing non-NumPy objects.

    https://github.com/telegraphic/hickle/issues/90"""

    # VisibleDeprecationWarning from newer numpy versions
    #np_array_data = np.array([[NESTED_DICT], ('What is this?',), {1, 2, 3, 7, 1}])
    arr = np.array([NESTED_DICT])#, ('What is this?',), {1, 2, 3, 7, 1}])
    dump(arr, test_file_name,**compression_kwargs)
    arr_hkl = load(test_file_name)
    assert np.all(arr == arr_hkl)

    arr2 = np.array(NESTED_DICT)
    dump(arr2, test_file_name,**compression_kwargs)
    arr_hkl2 = load(test_file_name)
    assert np.all(arr2 == arr_hkl2)


def test_string_numpy(test_file_name,compression_kwargs):
    """ Dumping and loading NumPy arrays containing Python 3 strings. """

    arr = np.array(["1313e", "was", "maybe?", "here"])
    dump(arr, test_file_name,**compression_kwargs)
    arr_hkl = load(test_file_name)
    assert np.all(arr == arr_hkl)


def test_list_object_numpy(test_file_name,compression_kwargs):
    """ Dumping and loading a list of NumPy arrays with objects.

    https://github.com/telegraphic/hickle/issues/90"""

    # VisibleDeprecationWarning from newer numpy versions
    lst = [np.array(NESTED_DICT)]#, np.array([('What is this?',),
                                 #           {1, 2, 3, 7, 1}])]
    dump(lst, test_file_name,**compression_kwargs)
    lst_hkl = load(test_file_name)
    assert np.all(lst[0] == lst_hkl[0])
    #assert np.all(lst[1] == lst_hkl[1])


def test_dict(test_file_name,compression_kwargs):
    """ Test dictionary dumping and loading """
    mode = 'w'

    dd = {
        'name': b'Danny',
        'age': 28,
        'height': 6.1,
        'dork': True,
        'nums': [1, 2, 3],
        'narr': np.array([1, 2, 3]),
    }

    dump(dd, test_file_name, mode,**compression_kwargs)
    dd_hkl = load(test_file_name)

    for k in dd.keys():
        try:
            assert k in dd_hkl.keys()

            if isinstance(dd[k], np.ndarray):
                assert np.all((dd[k], dd_hkl[k]))
            else:
                pass
            assert isinstance(dd_hkl[k], dd[k].__class__)
        except AssertionError:
            print(k)
            print(dd_hkl[k])
            print(dd[k])
            print(type(dd_hkl[k]), type(dd[k]))
            raise


def test_odict(test_file_name,compression_kwargs):
    """ Test ordered dictionary dumping and loading

    https://github.com/telegraphic/hickle/issues/65"""
    mode = 'w'

    od = odict(((3, [3, 0.1]), (7, [5, 0.1]), (5, [3, 0.1])))
    dump(od, test_file_name, mode,**compression_kwargs)
    od_hkl = load(test_file_name)

    assert od.keys() == od_hkl.keys()

    for od_item, od_hkl_item in zip(od.items(), od_hkl.items()):
        assert od_item == od_hkl_item


def test_empty_dict(test_file_name,compression_kwargs):
    """ Test empty dictionary dumping and loading

    https://github.com/telegraphic/hickle/issues/91"""
    mode = 'w'

    dump({}, test_file_name, mode,**compression_kwargs)
    assert load(test_file_name) == {}



# TODO consider converting to parameterized test
#      or enable implicit parameterizing of all tests
#      though compression_kwargs fixture providing
#      various combinations of compression and chunking
#      related keywords
@pytest.mark.no_compression
def test_compression(test_file_name):
    """ Test compression on datasets"""

    mode = 'w'
    dtypes = ['int32', 'float32', 'float64', 'complex64', 'complex128']

    comps = [None, 'gzip', 'lzf']

    for dt in dtypes:
        for cc in comps:
            array_obj = np.ones(32768, dtype=dt)
            dump(array_obj, test_file_name, mode, compression=cc)
            print(cc, os.path.getsize(test_file_name))
            array_hkl = load(test_file_name)
    try:
        assert array_hkl.dtype == array_obj.dtype
        assert np.all((array_hkl, array_obj))
    except AssertionError:
        print(array_hkl)
        print(array_obj)
        raise


def test_dict_int_key(test_file_name,compression_kwargs):
    """ Test for dictionaries with integer keys """
    mode = 'w'

    dd = {
        0: "test",
        1: "test2"
    }

    dump(dd, test_file_name, mode,**compression_kwargs)
    load(test_file_name)


def test_dict_nested(test_file_name,compression_kwargs):
    """ Test for dictionaries with integer keys """
    mode = 'w'

    dd = NESTED_DICT

    dump(dd, test_file_name, mode,**compression_kwargs)
    dd_hkl = load(test_file_name)

    ll_hkl = dd_hkl["level1_3"]["level2_1"]["level3_1"]
    ll = dd["level1_3"]["level2_1"]["level3_1"]
    assert ll == ll_hkl


def test_masked_dict(test_file_name,compression_kwargs):
    """ Test dictionaries with masked arrays """

    filename, mode = 'test.h5', 'w'

    dd = {
        "data": np.ma.array([1, 2, 3], mask=[True, False, False]),
        "data2": np.array([1, 2, 3, 4, 5])
    }

    dump(dd, test_file_name, mode,**compression_kwargs)
    dd_hkl = load(test_file_name)

    for k in dd.keys():
        try:
            assert k in dd_hkl.keys()
            if isinstance(dd[k], np.ndarray):
                assert np.all((dd[k], dd_hkl[k]))
            elif isinstance(dd[k], np.ma.MaskedArray):
                print(dd[k].data)
                print(dd_hkl[k].data)
                assert np.allclose(dd[k].data, dd_hkl[k].data)
                assert np.allclose(dd[k].mask, dd_hkl[k].mask)

            assert isinstance(dd_hkl[k], dd[k].__class__)

        except AssertionError:
            print(k)
            print(dd_hkl[k])
            print(dd[k])
            print(type(dd_hkl[k]), type(dd[k]))
            raise


def test_np_float(test_file_name,compression_kwargs):
    """ Test for singular np dtypes """
    mode = 'w'

    dtype_list = (np.float16, np.float32, np.float64,
                  np.complex64, np.complex128,
                  np.int8, np.int16, np.int32, np.int64,
                  np.uint8, np.uint16, np.uint32, np.uint64)

    for dt in dtype_list:

        dd = dt(1)
        dump(dd, test_file_name, mode,**compression_kwargs)
        dd_hkl = load(test_file_name)
        assert dd == dd_hkl
        assert dd.dtype == dd_hkl.dtype

    dd = {}
    for dt in dtype_list:
        dd[str(dt)] = dt(1.0)
    dump(dd, test_file_name, mode,**compression_kwargs)
    dd_hkl = load(test_file_name)

    print(dd)
    for dt in dtype_list:
        assert dd[str(dt)] == dd_hkl[str(dt)]


# TODO consider converting to parameterized test
#      or enable implicit parameterizing of all tests
#      though compression_kwargs fixture providing
#      various combinations of compression and chunking
#      related keywords
@pytest.mark.no_compression
def test_comp_kwargs(test_file_name):
    """ Test compression with some kwargs for shuffle and chunking """

    mode = 'w'
    dtypes = ['int32', 'float32', 'float64', 'complex64', 'complex128']

    comps = [None, 'gzip', 'lzf']
    chunks = [(100, 100), (250, 250)]
    shuffles = [True, False]
    scaleoffsets = [0, 1, 2]

    for dt in dtypes:
        for cc in comps:
            for ch in chunks:
                for sh in shuffles:
                    for so in scaleoffsets:
                        kwargs = {
                            'compression': cc,
                            'dtype': dt,
                            'chunks': ch,
                            'shuffle': sh,
                            'scaleoffset': so
                        }
                        array_obj = NESTED_DICT
                        dump(array_obj, test_file_name, mode, compression=cc)
                        print(kwargs, os.path.getsize(test_file_name))
                        load(test_file_name)


def test_list_numpy(test_file_name,compression_kwargs):
    """ Test converting a list of numpy arrays """

    mode = 'w'

    a = np.ones(1024)
    b = np.zeros(1000)
    c = [a, b]

    dump(c, test_file_name, mode,**compression_kwargs)
    dd_hkl = load(test_file_name)

    print(dd_hkl)

    assert isinstance(dd_hkl, list)
    assert isinstance(dd_hkl[0], np.ndarray)


def test_tuple_numpy(test_file_name,compression_kwargs):
    """ Test converting a list of numpy arrays """

    mode = 'w'

    a = np.ones(1024)
    b = np.zeros(1000)
    c = (a, b, a)

    dump(c, test_file_name, mode,**compression_kwargs)
    dd_hkl = load(test_file_name)

    print(dd_hkl)

    assert isinstance(dd_hkl, tuple)
    assert isinstance(dd_hkl[0], np.ndarray)


def test_numpy_dtype(test_file_name,compression_kwargs):
    """ Dumping and loading a NumPy dtype """

    dtype = np.dtype('float16')
    dump(dtype, test_file_name,**compression_kwargs)
    dtype_hkl = load(test_file_name)
    assert dtype == dtype_hkl


def test_none(test_file_name,compression_kwargs):
    """ Test None type hickling """

    mode = 'w'

    a = None

    dump(a, test_file_name, mode,**compression_kwargs)
    dd_hkl = load(test_file_name)
    print(a)
    print(dd_hkl)

    assert isinstance(dd_hkl, type(None))


def test_list_order(test_file_name,compression_kwargs):
    """ https://github.com/telegraphic/hickle/issues/26 """
    d = [np.arange(n + 1) for n in range(20)]
    dump(d, test_file_name,**compression_kwargs)
    d_hkl = load(test_file_name)

    try:
        for ii, xx in enumerate(d):
            assert d[ii].shape == d_hkl[ii].shape
        for ii, xx in enumerate(d):
            assert np.allclose(d[ii], d_hkl[ii])
    except AssertionError:
        print(d[ii], d_hkl[ii])
        raise


def test_embedded_array(test_file_name,compression_kwargs):
    """ See https://github.com/telegraphic/hickle/issues/24 """

    d_orig = [[np.array([10., 20.]), np.array([10, 20, 30])],
              [np.array([10, 2]), np.array([1.])]]
    dump(d_orig, test_file_name,**compression_kwargs)
    d_hkl = load(test_file_name)

    for ii, xx in enumerate(d_orig):
        for jj, yy in enumerate(xx):
            assert np.allclose(d_orig[ii][jj], d_hkl[ii][jj])

    print(d_hkl)
    print(d_orig)


##############
# NEW TESTS  #
###############
def generate_nested():
    a = [1, 2, 3]
    b = [a, a, a]
    c = [a, b, 's']
    d = [a, b, c, c, a]
    e = [d, d, d, d, 1]
    f = {'a': a, 'b': b, 'e': e}
    g = {'f': f, 'a': e, 'd': d}
    h = {'h': g, 'g': f}
    z = [f, a, b, c, d, e, f, g, h, g, h]
    a = np.array([1, 2, 3, 4])
    b = set([1, 2, 3, 4, 5])
    c = (1, 2, 3, 4, 5)
    d = np.ma.array([1, 2, 3, 4, 5, 6, 7, 8])
    z = {'a': a, 'b': b, 'c': c, 'd': d, 'z': z}
    return z

def test_dump_nested(test_file_name,compression_kwargs):
    """ Dump a complicated nested object to HDF5
    """
    z = generate_nested()
    dump(z, test_file_name, mode='w',**compression_kwargs)

def test_ndarray(test_file_name,compression_kwargs):
    a = np.array([1, 2, 3])
    b = np.array([2, 3, 4])
    z = (a, b)

    print("Original:")
    pprint(z)
    dump(z, test_file_name, mode='w',**compression_kwargs)

    print("\nReconstructed:")
    z = load(test_file_name)
    pprint(z)


def test_ndarray_masked(test_file_name,compression_kwargs):
    a = np.ma.array([1, 2, 3])
    b = np.ma.array([2, 3, 4], mask=[True, False, True])
    z = (a, b)

    print("Original:")
    pprint(z)
    dump(z, test_file_name, mode='w',**compression_kwargs)

    print("\nReconstructed:")
    z = load(test_file_name)
    pprint(z)


def test_simple_dict(test_file_name,compression_kwargs):
    a = {'key1': 1, 'key2': 2}

    dump(a, test_file_name,**compression_kwargs)
    z = load(test_file_name)

    pprint(a)
    pprint(z)


def test_complex_dict(test_file_name,compression_kwargs):
    a = {'akey': 1, 'akey2': 2}
    c = {'ckey': "hello", "ckey2": "hi there"}
    z = {'zkey1': a, 'zkey2': a, 'zkey3': c}

    print("Original:")
    pprint(z)
    dump(z, test_file_name, mode='w',**compression_kwargs)

    print("\nReconstructed:")
    z = load(test_file_name)
    pprint(z)

def test_complex(test_file_name,compression_kwargs):
    """ Test complex value dtype is handled correctly

    https://github.com/telegraphic/hickle/issues/29 """

    data = {"A": 1.5, "B": 1.5 + 1j, "C": np.linspace(0, 1, 4) + 2j}
    dump(data, test_file_name,**compression_kwargs)
    data2 = load(test_file_name)
    for key in data.keys():
        assert isinstance(data[key], data2[key].__class__)


def test_nonstring_keys(test_file_name,compression_kwargs):
    """ Test that keys are reconstructed back to their original datatypes
    https://github.com/telegraphic/hickle/issues/36
    """

    data = {
            u'test': 123,
            'def': [b'test'],
            'hik': np.array([1, 2, 3]),
            0: 0,
            True: ['test'],
            1.1: 'hey',
            1j: 'complex_hashable',
            (1, 2): 'boo',
            ('A', 17.4, 42): [1, 7, 'A'],
            (): '1313e was here',
            '0': 0,
            None: None
            }

    print(data)
    dump(data, test_file_name,**compression_kwargs)
    data2 = load(test_file_name)
    print(data2)

    for key in data.keys():
        assert key in data2.keys()

    print(data2)

@pytest.mark.no_compression
def test_scalar_compression(test_file_name):
    """ Test bug where compression causes a crash on scalar datasets

    (Scalars are incompressible!)
    https://github.com/telegraphic/hickle/issues/37
    """
    data = {'a': 0, 'b': np.float(2), 'c': True}

    dump(data, test_file_name, compression='gzip')
    data2 = load(test_file_name)

    print(data2)
    for key in data.keys():
        assert isinstance(data[key], data2[key].__class__)


def test_bytes(test_file_name,compression_kwargs):
    """ Dumping and loading a string. PYTHON3 ONLY """

    mode = 'w'
    string_obj = b"The quick brown fox jumps over the lazy dog"
    dump(string_obj, test_file_name, mode,**compression_kwargs)
    string_hkl = load(test_file_name)
    print(type(string_obj))
    print(type(string_hkl))
    assert isinstance(string_hkl, bytes)
    assert string_obj == string_hkl


def test_np_scalar(test_file_name,compression_kwargs):
    """ Numpy scalar datatype

    https://github.com/telegraphic/hickle/issues/50
    """

    r0 = {'test': np.float64(10.)}
    dump(r0, test_file_name,**compression_kwargs)
    r = load(test_file_name)
    print(r)
    assert isinstance(r0['test'], r['test'].__class__)


def test_slash_dict_keys(test_file_name,compression_kwargs):
    """ Support for having slashes in dict keys

    https://github.com/telegraphic/hickle/issues/124"""
    dct = {'a/b': [1, '2'], 1.4: 3}

    dump(dct, test_file_name, 'w',**compression_kwargs)
    dct_hkl = load(test_file_name)

    assert isinstance(dct_hkl, dict)
    for key, val in dct_hkl.items():
        assert val == dct.get(key)

    # Check that having backslashes in dict keys will serialize the dict
    dct2 = {'a\\b': [1, '2'], 1.4: 3}
    with pytest.warns(None) as not_expected:
        dump(dct2, test_file_name,**compression_kwargs)
    assert not not_expected


# %% MAIN SCRIPT
if __name__ == '__main__':
    """ Some tests and examples """
    from _pytest.fixtures import FixtureRequest

    for filename in test_file_name(FixtureRequest(test_np_scalar)):
        test_np_scalar(filename)
    for filename in test_file_name(FixtureRequest(test_scalar_compression)):
        test_scalar_compression(filename)
    for filename in test_file_name(FixtureRequest(test_complex)):
        test_complex(filename)
    for filename in test_file_name(FixtureRequest(test_none)):
        test_none(filename)
    for filename in test_file_name(FixtureRequest(test_masked_dict)):
        test_masked_dict(filename)
    for filename in test_file_name(FixtureRequest(test_list)):
        test_list(filename)
    for filename in test_file_name(FixtureRequest(test_set)):
        test_set(filename)
    for filename in test_file_name(FixtureRequest(test_numpy)):
        test_numpy(filename)
    for filename in test_file_name(FixtureRequest(test_dict)):
        test_dict(filename)
    for filename in test_file_name(FixtureRequest(test_odict)):
        test_odict(filename)
    for filename in test_file_name(FixtureRequest(test_empty_dict)):
        test_empty_dict(filename)
    for filename in test_file_name(FixtureRequest(test_compression)):
        test_compression(filename)
    for filename in test_file_name(FixtureRequest(test_masked)):
        test_masked(filename)
    for filename in test_file_name(FixtureRequest(test_dict_nested)):
        test_dict_nested(filename)
    for filename in test_file_name(FixtureRequest(test_comp_kwargs)):
        test_comp_kwargs(filename)
    for filename in test_file_name(FixtureRequest(test_list_numpy)):
        test_list_numpy(filename)
    for filename in test_file_name(FixtureRequest(test_tuple_numpy)):
        test_tuple_numpy(filename)
    for filename in test_file_name(FixtureRequest(test_list_order)):
        test_list_order(filename)
    for filename in test_file_name(FixtureRequest(test_embedded_array)):
        test_embedded_array(filename)
    for filename in test_file_name(FixtureRequest(test_np_float)):
        test_np_float(filename)
    for filename in test_file_name(FixtureRequest(test_string)):
        test_string(filename)
    for filename in test_file_name(FixtureRequest(test_nonstring_keys)):
        test_nonstring_keys(filename)
    for filename in test_file_name(FixtureRequest(test_bytes)):
        test_bytes(filename)

    # NEW TESTS
    for filename in test_file_name(FixtureRequest(test_dump_nested)):
        test_dump_nested(filename)
    for filename in test_file_name(FixtureRequest(test_ndarray)):
        test_ndarray(filename)
    for filename in test_file_name(FixtureRequest(test_ndarray_masked)):
        test_ndarray_masked(filename)
    for filename in test_file_name(FixtureRequest(test_simple_dict)):
        test_simple_dict(filename)
    for filename in test_file_name(FixtureRequest(test_complex_dict)):
        test_complex_dict(filename)
    for filename in test_file_name(FixtureRequest(test_dict_int_key)):
        test_dict_int_key(filename)
    for filename in test_file_name(FixtureRequest(test_local_func)):
        test_local_func(filename)
    for filename in test_file_name(FixtureRequest(test_slash_dict_keys)):
        test_slash_dict_keys(filename)
    test_invalid_file()
    for filename in test_file_name(FixtureRequest(test_non_empty_group)):
        test_non_empty_group(filename)
    for filename in test_file_name(FixtureRequest(test_numpy_dtype)):
        test_numpy_dtype(filename)
    for filename in test_file_name(FixtureRequest(test_object_numpy)):
        test_object_numpy(filename)
    for filename in test_file_name(FixtureRequest(test_string_numpy)):
        test_string_numpy(filename)
    for filename in test_file_name(FixtureRequest(test_list_object_numpy)):
        test_list_object_numpy(filename)

    # Cleanup
    for filename in test_file_name(FixtureRequest(print)):
        print(filename)
