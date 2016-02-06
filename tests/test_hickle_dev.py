from hickle import check_is_iterable, check_iterable_item_type, _dump, dump, load, sort_keys
import h5py
import numpy as np
from pprint import pprint

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
    b = {1, 2, 3, 4, 5}
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

    a = {1, 2, 3, 4}
    b = {5, 6, 7, 8}
    c = {9, 10, 11, 12}
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


if __name__ == "__main__":
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
    print("OK")