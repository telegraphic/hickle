#! /usr/bin/env python
# encoding: utf-8
"""
# test_hickle.py

Unit tests for hickle module.

"""

import os
from hickle import *
import hickle
import hashlib
import time

import h5py
import numpy as np
from pprint import pprint

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

DUMP_CACHE = []             # Used in test_track_times()


def test_string():
    """ Dumping and loading a string """
    filename, mode = 'test.h5', 'w'
    string_obj = "The quick brown fox jumps over the lazy dog"
    dump(string_obj, filename, mode)
    string_hkl = load(filename)
    #print "Initial list:   %s"%list_obj
    #print "Unhickled data: %s"%list_hkl
    try:
        assert type(string_obj) == type(string_hkl) == str
        assert string_obj == string_hkl
        os.remove(filename)
    except AssertionError:
        os.remove(filename)
        raise


def test_unicode():
    """ Dumping and loading a unicode string """
    filename, mode = 'test.h5', 'w'
    u = unichr(233) + unichr(0x0bf2) + unichr(3972) + unichr(6000)
    dump(u, filename, mode)
    u_hkl = load(filename)

    try:
        assert type(u) == type(u_hkl) == unicode
        assert u == u_hkl
        # For those interested, uncomment below to see what those codes are:
        # for i, c in enumerate(u_hkl):
        #     print i, '%04x' % ord(c), unicodedata.category(c),
        #     print unicodedata.name(c)
    except AssertionError:
        os.remove(filename)
        raise


def test_list():
    """ Dumping and loading a list """
    filename, mode = 'test.h5', 'w'
    list_obj = [1, 2, 3, 4, 5]
    dump(list_obj, filename, mode)
    list_hkl = load(filename)
    #print "Initial list:   %s"%list_obj
    #print "Unhickled data: %s"%list_hkl
    try:
        assert type(list_obj) == type(list_hkl) == list
        assert list_obj == list_hkl
        import h5py
        a = h5py.File(filename)

        os.remove(filename)
    except AssertionError:
        print "ERR:", list_obj, list_hkl
        import h5py
        os.remove(filename)
        raise


def test_set():
    """ Dumping and loading a list """
    filename, mode = 'test.h5', 'w'
    list_obj = set([1, 0, 3, 4.5, 11.2])
    dump(list_obj, filename, mode)
    list_hkl = load(filename)
    #print "Initial list:   %s"%list_obj
    #print "Unhickled data: %s"%list_hkl
    try:
        assert type(list_obj) == type(list_hkl) == set
        assert list_obj == list_hkl
        os.remove(filename)
    except AssertionError:
        os.remove(filename)
        raise


def test_numpy():
    """ Dumping and loading numpy array """
    filename, mode = 'test.h5', 'w'
    dtypes = ['float32', 'float64', 'complex64', 'complex128']
    
    for dt in dtypes:
        array_obj = np.ones(8, dtype=dt)
        dump(array_obj, filename, mode)
        array_hkl = load(filename)
    try:
        assert array_hkl.dtype == array_obj.dtype
        assert np.all((array_hkl, array_obj))
        os.remove(filename)
    except AssertionError:
        os.remove(filename)
        print array_hkl
        print array_obj
        raise


def test_masked():
    """ Test masked numpy array """
    filename, mode = 'test.h5', 'w'    
    a = np.ma.array([1,2,3,4], dtype='float32', mask=[0,1,0,0])
    
    dump(a, filename, mode)
    a_hkl = load(filename)
    
    try:
        assert a_hkl.dtype == a.dtype
        assert np.all((a_hkl, a))
        os.remove(filename)
    except AssertionError:
        os.remove(filename)
        print a_hkl
        print a
        raise
        

def test_dict():
    """ Test dictionary dumping and loading """
    filename, mode = 'test.h5', 'w'
    
    dd = {
        'name'   : 'Danny',
        'age'    : 28,
        'height' : 6.1,
        'dork'   : True,
        'nums'   : [1, 2, 3],
        'narr'   : np.array([1,2,3]),
        #'unic'   : u'dan[at]thetelegraphic.com'
    }
    
    
    dump(dd, filename, mode)
    dd_hkl = load(filename)
    
    for k in dd.keys():
        try:
            assert k in dd_hkl.keys()
            
            if type(dd[k]) is type(np.array([1])):
                assert np.all((dd[k], dd_hkl[k]))
            else:
                #assert dd_hkl[k] == dd[k]
                pass
            assert type(dd_hkl[k]) == type(dd[k])
        except AssertionError:
            print k
            print dd_hkl[k]
            print dd[k]
            print type(dd_hkl[k]), type(dd[k])
            os.remove(filename)
            raise
    os.remove(filename)


def test_compression():
    """ Test compression on datasets"""
    
    filename, mode = 'test.h5', 'w'
    dtypes = ['int32', 'float32', 'float64', 'complex64', 'complex128']
    
    comps = [None, 'gzip', 'lzf']
    
    for dt in dtypes:
        for cc in comps:
            array_obj = np.ones(32768, dtype=dt)
            dump(array_obj, filename, mode, compression=cc)
            print cc, os.path.getsize(filename)
            array_hkl = load(filename)
    try:
        assert array_hkl.dtype == array_obj.dtype
        assert np.all((array_hkl, array_obj))
        os.remove(filename)
    except AssertionError:
        os.remove(filename)
        print array_hkl
        print array_obj
        raise


def test_dict_int_key():
    """ Test for dictionaries with integer keys """
    filename, mode = 'test.h5', 'w'

    dd = {
        0: "test",
        1: "test2"
    }

    dump(dd, filename, mode)
    dd_hkl = load(filename)

    
    os.remove(filename)


def test_dict_nested():
    """ Test for dictionaries with integer keys """
    filename, mode = 'test.h5', 'w'

    dd = NESTED_DICT

    dump(dd, filename, mode)
    dd_hkl = load(filename)
    
    ll_hkl = dd_hkl["level1_3"]["level2_1"]["level3_1"]
    ll     = dd["level1_3"]["level2_1"]["level3_1"]
    assert ll == ll_hkl
    os.remove(filename)


def test_masked_dict():
    """ Test dictionaries with masked arrays """

    filename, mode = 'test.h5', 'w'

    dd = {
        "data"  : np.ma.array([1,2,3], mask=[True, False, False]),
        "data2" : np.array([1,2,3,4,5])
    }

    dump(dd, filename, mode)
    dd_hkl = load(filename)
    
    for k in dd.keys():
        try:
            assert k in dd_hkl.keys()
            if type(dd[k]) is type(np.array([1])):
                assert np.all((dd[k], dd_hkl[k]))
            elif type(dd[k]) is type(np.ma.array([1])):
                print dd[k].data
                print dd_hkl[k].data
                assert np.allclose(dd[k].data, dd_hkl[k].data)
                assert np.allclose(dd[k].mask, dd_hkl[k].mask)
                
            assert type(dd_hkl[k]) == type(dd[k])

        except AssertionError:
            print k
            print dd_hkl[k]
            print dd[k]
            print type(dd_hkl[k]), type(dd[k])
            os.remove(filename)
            raise
    os.remove(filename)


def test_nomatch():
    """ Test for non-supported data types.

     Note: don't remember what I was trying to do with this test.
     Ignoring it for now.
     """
    filename, mode = 'nomatch.h5', 'w'

    dd = Exception('Nothing to see here')
    no_match = False
    dump(dd, filename, mode)
    
    #dd_hkl = load(filename)
    dd_hkl = load(filename, safe=False)
    
    assert type(dd_hkl) == type(dd) == Exception
    os.remove(filename)


def test_np_float():
    """ Test for singular np dtypes """
    filename, mode = 'np_float.h5', 'w'    
    
    dtype_list = (np.float16, np.float32, np.float64, 
                  np.complex64, np.complex128,
                  np.int8, np.int16, np.int32, np.int64,
                  np.uint8, np.uint16, np.uint32, np.uint64)
                  
    for dt in dtype_list:
    
        dd = dt(1)
        dump(dd, filename, mode)
        dd_hkl = load(filename)
        assert dd == dd_hkl
        assert dd.dtype == dd_hkl.dtype
        os.remove(filename)

    dd = {}
    for dt in dtype_list:
        dd[str(dt)] = dt(1.0)
    dump(dd, filename, mode)
    dd_hkl = load(filename)

    print dd
    for dt in dtype_list:
        assert dd[str(dt)] == dd_hkl[str(dt)]

    os.remove(filename)


def md5sum(filename, blocksize=65536):
    """ Compute MD5 sum for a given file """
    hash = hashlib.md5()
                
    with open(filename, "r+b") as f:
        for block in iter(lambda: f.read(blocksize), ""):
            hash.update(block)
    return hash.hexdigest()


def caching_dump(obj, filename, *args, **kwargs):
    """ Save arguments of all dump calls """
    DUMP_CACHE.append((obj, filename, args, kwargs))
    return hickle_dump(obj, filename, *args, **kwargs)


def test_track_times():
    """ Verify that track_times = False produces identical files """
    hashes = []
    for obj, filename, mode, kwargs in DUMP_CACHE:
        if isinstance(filename, hickle.H5FileWrapper):
            filename = str(filename.file_name)
        kwargs['track_times'] = False
        caching_dump(obj, filename, mode, **kwargs)
        hashes.append(md5sum(filename))
        os.remove(filename)

    time.sleep(1)

    for hash1, (obj, filename, mode, kwargs) in zip(hashes, DUMP_CACHE):
        if isinstance(filename, hickle.H5FileWrapper):
            filename = str(filename.file_name)
        caching_dump(obj, filename, mode, **kwargs)
        hash2 = md5sum(filename)
        print hash1, hash2
        try:
            assert hash1 == hash2
            os.remove(filename)
        except AssertionError:
            os.remove(filename)
            raise


def test_comp_kwargs():
    """ Test compression with some kwargs for shuffle and chunking """

    filename, mode = 'test.h5', 'w'
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
                            'compression' : cc,
                            'dtype': dt,
                            'chunks': ch,
                            'shuffle': sh,
                            'scaleoffset': so
                        }
                        #array_obj = np.random.random_integers(low=-8192, high=8192, size=(1000, 1000)).astype(dt)
                        array_obj = NESTED_DICT
                        dump(array_obj, filename, mode, compression=cc)
                        print kwargs, os.path.getsize(filename)
                        array_hkl = load(filename)
    try:
        os.remove(filename)
    except AssertionError:
        os.remove(filename)
        print array_hkl
        print array_obj
        raise


def test_list_numpy():
    """ Test converting a list of numpy arrays """

    filename, mode = 'test.h5', 'w'

    a = np.ones(1024)
    b = np.zeros(1000)
    c = [a, b]

    dump(c, filename, mode)
    dd_hkl = load(filename)

    print dd_hkl

    assert isinstance(dd_hkl, list)
    assert isinstance(dd_hkl[0], np.ndarray)


    os.remove(filename)


def test_tuple_numpy():
    """ Test converting a list of numpy arrays """

    filename, mode = 'test.h5', 'w'

    a = np.ones(1024)
    b = np.zeros(1000)
    c = (a, b, a)

    dump(c, filename, mode)
    dd_hkl = load(filename)

    print dd_hkl

    assert isinstance(dd_hkl, tuple)
    assert isinstance(dd_hkl[0], np.ndarray)


    os.remove(filename)


def test_none():
    """ Test None type hickling """
    
    filename, mode = 'test.h5', 'w'

    a = None

    dump(a, filename, mode)
    dd_hkl = load(filename)
    print a
    print dd_hkl

    assert isinstance(dd_hkl, NoneType)

    os.remove(filename)


def test_dict_none():
    """ Test None type hickling """
    
    filename, mode = 'test.h5', 'w'

    a = {'a': 1, 'b' : None}

    dump(a, filename, mode)
    dd_hkl = load(filename)
    print a
    print dd_hkl

    assert isinstance(a['b'], NoneType)

    os.remove(filename)


def test_file_open_close():
    """ https://github.com/telegraphic/hickle/issues/20 """
    try:
        import h5py
        f = h5py.File('test.hdf', 'w')
        a = np.arange(5)
    
        dump(a, 'test.hkl')
        dump(a, 'test.hkl')
    
        dump(a, f, mode='w')
        try:
            dump(a, f, mode='w')
        except ClosedFileError:
            print "Tests: Closed file exception caught"
        
    finally:
        os.remove('test.hdf')
        os.remove('test.hkl')


def run_file_cleanup():
    """ Clean up temp files """
    for filename in ('test.hdf', 'test.hkl', 'test.h5'):
        try:
            os.remove(filename)
        except OSError:
            pass


def test_list_long_type():
    """ Check long comes back out as a long """
    filename, mode = 'test.h5', 'w'
    list_obj = [1L, 2L, 3L, 4L, 5L]
    dump(list_obj, filename, mode)
    list_hkl = load(filename)
    #print "Initial list:   %s"%list_obj
    #print "Unhickled data: %s"%list_hkl
    try:
        assert type(list_obj) == type(list_hkl) == list
        assert list_obj == list_hkl
        assert type(list_obj[0]) == type(list_hkl[0])
        
        os.remove(filename)
    except AssertionError:
        print "ERR:", list_obj, list_hkl
        import h5py
        a = h5py.File(filename)
        print a.keys()
        print a['data'].keys()
        os.remove(filename)
        raise


def test_list_order():
    """ https://github.com/telegraphic/hickle/issues/26 """
    d = [np.arange(n + 1) for n in range(20)]
    hickle.dump(d, 'test.h5')
    d_hkl = hickle.load('test.h5')
    
    try:
        for ii, xx in enumerate(d):
            assert d[ii].shape == d_hkl[ii].shape
        for ii, xx in enumerate(d):
            assert np.allclose(d[ii], d_hkl[ii])
    except AssertionError:
        print d[ii], d_hkl[ii]
        raise


def test_embedded_array():
    """ See https://github.com/telegraphic/hickle/issues/24 """
    
    d_orig = [[np.array([10., 20.]), np.array([10, 20, 30])], [np.array([10, 2]), np.array([1.])]]
    hickle.dump(d_orig, 'test.h5')
    d_hkl = hickle.load('test.h5')
    
    for ii, xx in enumerate(d_orig):
        for jj, yy in enumerate(xx):
            assert np.allclose(d_orig[ii][jj], d_hkl[ii][jj])
    
    print d_hkl
    print d_orig


################
## NEW TESTS  ##
################


def generate_nested():
    a = [1, 2, 3]
    b = [a, a, a]
    c = [a, b, 's']
    d = [a, b, c, c, a]
    e = [d, d, d, d, 1]
    f = {'a' : a, 'b' : b, 'e' : e}
    g = {'f' : f, 'a' : e, 'd': d}
    h = {'h': g, 'g' : f}
    z = [f, a, b, c, d, e, f, g, h, g, h]
    a = np.array([1, 2, 3, 4])
    b = set([1, 2, 3, 4, 5])
    c = (1, 2, 3, 4, 5)
    d = np.ma.array([1, 2, 3, 4, 5, 6, 7, 8])
    z = {'a': a, 'b': b, 'c': c, 'd': d, 'z': z}
    return z


def test_is_iterable():
    a = [1, 2, 3]
    b = 1

    assert check_is_iterable(a) == True
    assert check_is_iterable(b) == False


def test_check_iterable_item_type():

    a = [1, 2, 3]
    b = [a, a, a]
    c = [a, b, 's']

    type_a = check_iterable_item_type(a)
    type_b = check_iterable_item_type(b)
    type_c = check_iterable_item_type(c)

    assert type_a is int
    assert type_b is list
    assert type_c == False


def test_dump_nested():
    """ Dump a complicated nested object to HDF5
    """
    z = generate_nested()
    dump(z, 'test.hkl', mode='w')


def test_load():

    a = set([1, 2, 3, 4])
    b = set([5, 6, 7, 8])
    c = set([9, 10, 11, 12])
    z = (a, b, c)
    z = [z, z]
    z = (z, z, z, z, z)

    print "Original:"
    pprint(z)
    dump(z, 'test.hkl', mode='w')

    print "\nReconstructed:"
    z = load('test.hkl')
    pprint(z)


def test_sort_keys():
    keys = ['data_0', 'data_1', 'data_2', 'data_3', 'data_10']
    keys_sorted = ['data_0', 'data_1', 'data_2', 'data_3', 'data_10']
    assert sort_keys(keys) == keys_sorted


def test_ndarray():

    a = np.array([1,2,3])
    b = np.array([2,3,4])
    z = (a, b)

    print "Original:"
    pprint(z)
    dump(z, 'test.hkl', mode='w')

    print "\nReconstructed:"
    z = load('test.hkl')
    pprint(z)


def test_ndarray_masked():

    a = np.ma.array([1,2,3])
    b = np.ma.array([2,3,4], mask=[True, False, True])
    z = (a, b)

    print "Original:"
    pprint(z)
    dump(z, 'test.hkl', mode='w')

    print "\nReconstructed:"
    z = load('test.hkl')
    pprint(z)


def test_simple_dict():
    a = {'key1': 1, 'key2': 2}

    dump(a, 'test.hkl')
    z = load('test.hkl')

    pprint(a)
    pprint(z)


def test_complex_dict():
    a = {'akey': 1, 'akey2': 2}
    b = {'bkey': 2.0, 'bkey3': long(3.0)}
    c = {'ckey': "hello", "ckey2": "hi there"}
    z = {'zkey1': a, 'zkey2': b, 'zkey3': c}

    print "Original:"
    pprint(z)
    dump(z, 'test.hkl', mode='w')

    print "\nReconstructed:"
    z = load('test.hkl')
    pprint(z)


def test_unicode():
    a = u"unicode test"
    dump(a, 'test.hkl', mode='w')

    z = load('test.hkl')
    assert a == z
    assert type(a) == type(z) == unicode
    pprint(z)


def test_legacy_hickles():

    try:
        a = load("hickle_1_1_0.hkl")
        b = load("hickle_1_3_0.hkl")
        
        import h5py
        d = h5py.File("hickle_1_1_0.hkl")["data"]["a"][:]
        d2 = h5py.File("hickle_1_3_0.hkl")["data"]["a"][:]
        assert np.allclose(d, a["a"])
        assert np.allclose(d2, b["a"])
        
    except IOError:
        # For travis-CI
        a = load("tests/hickle_1_1_0.hkl")
        b = load("tests/hickle_1_3_0.hkl")
    
    print a 
    print b


def test_multi_hickle():
    import os
    a = {'a': 123, 'b': [1, 2, 4]}

    if os.path.exists("test.hkl"):
        os.remove("test.hkl")
    dump(a, "test.hkl", path="/test", mode="w")
    dump(a, "test.hkl", path="/test2", mode="r+")
    dump(a, "test.hkl", path="/test3", mode="r+")
    dump(a, "test.hkl", path="/test4", mode="r+")

    a = load("test.hkl", path="/test")
    b = load("test.hkl", path="/test2")
    c = load("test.hkl", path="/test3")
    d = load("test.hkl", path="/test4")
    os.remove("test.hkl")

def test_complex():
    """ Test complex value dtype is handled correctly
    
    https://github.com/telegraphic/hickle/issues/29 """
    
    data = {"A":1.5, "B":1.5 + 1j, "C":np.linspace(0,1,4) + 2j}
    dump(data, "test.hkl")   
    data2 = load("test.hkl")
    for key in data.keys():
        assert type(data[key]) == type(data2[key])

    
if __name__ == '__main__':
    """ Some tests and examples """
    test_complex()
    test_file_open_close()
    test_dict_none()
    test_none()
    test_unicode()
    test_string()
    test_masked_dict()
    test_list()
    test_set()
    test_numpy()
    test_dict()
    test_compression()
    test_masked()
    test_dict_nested()
    test_comp_kwargs()
    test_list_numpy()
    test_tuple_numpy()
    test_track_times()
    test_list_order()
    test_embedded_array()
    test_np_float()

    # NEW TESTS
    test_legacy_hickles()
    test_is_iterable()
    test_check_iterable_item_type()
    test_dump_nested()
    test_load()
    test_sort_keys()
    test_ndarray()
    test_ndarray_masked()
    test_simple_dict()
    test_complex_dict()
    test_unicode()
    test_multi_hickle()

    #FAILING TESTS:
    #test_nomatch()
    #test_dict_int_key()
    #test_list_long_type()

    # Cleanup
    run_file_cleanup()
    print "ALL TESTS PASSED!"