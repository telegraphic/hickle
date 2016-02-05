from hickle_dev import check_is_iterable, check_iterable_item_type, _dump, dump, load
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
    z = [b, b, b] #, z]

    z = (z, z, z)
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

    z = {1, 2, 3, 4}
    z = (z, z, z)
    z = [z, z]
    z = (z, z, z, z, z)

    print "Original:"
    pprint(z)
    dump(z, 'test.hkl', mode='w')

    print "\nReconstructed:"
    z = load('test.hkl')
    pprint(z)

if __name__ == "__main__":
    test_is_iterable()
    test_check_iterable_item_type()
    test_dump_nested()
    test_load()
    print("OK")