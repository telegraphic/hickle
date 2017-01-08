import hickle as hkl
from astropy.units import Quantity
from astropy.constants import Constant, EMConstant, G
from astropy.table import Table

def test_astropy_quantity():

    for uu in ['m^3', 'm^3 / s', 'kg/pc']:
        a = Quantity(7, unit=uu)

        hkl.dump(a, "test.h5")
        b = hkl.load("test.h5")

        assert a == b
        assert a.unit == b.unit

        a *= a
        hkl.dump(a, "test.h5")
        b = hkl.load("test.h5")
        assert a == b
        assert a.unit == b.unit

def test_astropy_constant():
        hkl.dump(G, "test.h5")
        gg = hkl.load("test.h5")

        print G
        print gg

def test_astropy_table():
    t = Table([[1, 2], [3, 4]], names=('a', 'b'), meta={'name': 'test_thing'})

    hkl.dump(t, "test.h5")
    t2 = hkl.load("test.h5")

    print t
    print t.meta
    print t2
    print t2.meta

if __name__ == "__main__":
    test_astropy_quantity()
    test_astropy_constant()
    test_astropy_table()