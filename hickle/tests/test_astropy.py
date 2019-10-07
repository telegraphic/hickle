import hickle as hkl
from astropy.units import Quantity
from astropy.time import Time
from astropy.coordinates import Angle, SkyCoord
from astropy.constants import Constant, EMConstant, G
from astropy.table import Table
import numpy as np
from py.path import local

# Set the current working directory to the temporary directory
local.get_temproot().chdir()

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

def test_astropy_quantity_array():
    a = Quantity([1,2,3], unit='m')

    hkl.dump(a, "test_ap.h5")
    b = hkl.load("test_ap.h5")

    assert np.allclose(a.value, b.value)
    assert a.unit == b.unit

def test_astropy_time_array():
    times = ['1999-01-01T00:00:00.123456789', '2010-01-01T00:00:00']
    t1 = Time(times, format='isot', scale='utc')
    hkl.dump(t1, "test_ap2.h5")
    t2 = hkl.load("test_ap2.h5")

    print(t1)
    print(t2)
    assert t1.value.shape == t2.value.shape
    for ii in range(len(t1)):
        assert t1.value[ii] == t2.value[ii]
    assert t1.format == t2.format
    assert t1.scale == t2.scale

    times = [58264, 58265, 58266]
    t1 = Time(times, format='mjd', scale='utc')
    hkl.dump(t1, "test_ap2.h5")
    t2 = hkl.load("test_ap2.h5")

    print(t1)
    print(t2)
    assert t1.value.shape == t2.value.shape
    assert np.allclose(t1.value, t2.value)
    assert t1.format == t2.format
    assert t1.scale == t2.scale

def test_astropy_angle():
    for uu in ['radian', 'degree']:
        a = Angle(1.02, unit=uu)

        hkl.dump(a, "test_ap.h5")
        b = hkl.load("test_ap.h5")
        assert a == b
        assert a.unit == b.unit

def test_astropy_angle_array():
    a = Angle([1,2,3], unit='degree')

    hkl.dump(a, "test_ap.h5")
    b = hkl.load("test_ap.h5")

    assert np.allclose(a.value, b.value)
    assert a.unit == b.unit

def test_astropy_skycoord():
    ra = Angle(['1d20m', '1d21m'], unit='degree')
    dec = Angle(['33d0m0s', '33d01m'], unit='degree')
    radec = SkyCoord(ra, dec)
    hkl.dump(radec, "test_ap.h5")
    radec2 = hkl.load("test_ap.h5")
    assert np.allclose(radec.ra.value, radec2.ra.value)
    assert np.allclose(radec.dec.value, radec2.dec.value)

    ra = Angle(['1d20m', '1d21m'], unit='hourangle')
    dec = Angle(['33d0m0s', '33d01m'], unit='degree')
    radec = SkyCoord(ra, dec)
    hkl.dump(radec, "test_ap.h5")
    radec2 = hkl.load("test_ap.h5")
    assert np.allclose(radec.ra.value, radec2.ra.value)
    assert np.allclose(radec.dec.value, radec2.dec.value)

if __name__ == "__main__":
    test_astropy_quantity()
    #test_astropy_constant()
    test_astropy_table()
    test_astropy_quantity_array()
    test_astropy_time_array()
    test_astropy_angle()
    test_astropy_angle_array()
    test_astropy_skycoord()
