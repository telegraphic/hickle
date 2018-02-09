import hickle as hkl
import numpy as np
import os

def test_legacy_load():
    try:
        hkl.load("hickle_1_1_0.hkl")
    except RuntimeError:
        pass

def test_py2_load():
    pass

def test_string_handling():

    x = 'hi'
    hkl.dump(x, 'test.hkl')  # fine
    y = hkl.load('test.hkl')
    print(type(x), type(y))
    assert type(x) == type(y)

    x = u'hi'
    hkl.dump(x, 'test.hkl')  # fine, but a str is returned
    y = hkl.load('test.hkl')
    print(type(x), type(y))
    assert type(x) == type(y)#

    x = ['hello', 'hi', 'my_name_is']
    hkl.dump(x, 'test.hkl')  # works, but a bytes object is returned in the list
    y = hkl.load('test.hkl')
    print(x)
    print(y)
    print(type(x[0]), type(y[0]))
    assert type(x[0]) == type(y[0])

    x = [b'hello', b'hi', b'my_name_is']
    hkl.dump(x, 'test.hkl')  # works, but a bytes object is returned in the list
    y = hkl.load('test.hkl')
    print(x)
    print(y)
    print(type(x[0]), type(y[0]))
    assert type(x[0]) == type(y[0])

    os.remove('test.hkl')


if __name__ == "__main__":

    test_legacy_load()
    test_string_handling()


