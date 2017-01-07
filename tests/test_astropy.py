import hickle as hkl
from astropy.units import Quantity
from astropy.constants import Constant, EMConstant, G

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

if __name__ == "__main__":
    test_astropy_quantity()
    test_astropy_constant()