import hickle as hkl
from astropy.units import Quantity
from astropy.constants import Constant, EMConstant, G
from astropy.table import Table
import numpy as np

def test_astropy_quantity():

    for uu in ['m^3', 'm^3 / s', 'kg/pc']:
        a = Quantity(7, unit=uu)

        hkl.dump(a, "test_ap.h5")
        b = hkl.load("test_ap.h5")

        assert a == b
        assert a.unit == b.unit

        a *= a
        hkl.dump(a, "test_ap.h5")
        b = hkl.load("test_ap.h5")
        assert a == b
        assert a.unit == b.unit

def TODO_test_astropy_constant():
        hkl.dump(G, "test_ap.h5")
        gg = hkl.load("test_ap.h5")

        print(G)
        print(gg)

def test_astropy_table():
    t = Table([[1, 2], [3, 4]], names=('a', 'b'), meta={'name': 'test_thing'})

    hkl.dump({'a': t}, "test_ap.h5")
    t2 = hkl.load("test_ap.h5")['a']

    print(t)
    print(t.meta)
    print(t2)
    print(t2.meta)

    print(t.dtype, t2.dtype)
    assert t.meta == t2.meta
    assert t.dtype == t2.dtype

    assert np.allclose(t['a'].astype('float32'), t2['a'].astype('float32'))
    assert np.allclose(t['b'].astype('float32'), t2['b'].astype('float32'))

if __name__ == "__main__":
    test_astropy_quantity()
    #test_astropy_constant()
    test_astropy_table()
