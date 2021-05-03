#! /usr/bin/env python
# encoding: utf-8
"""
# test_load_astropy

Unit tests for hickle module -- astropy loader.

"""
# %% IMPORTS
# Package imports
import h5py as h5
import numpy as np
import pytest
from astropy.units import Quantity
from astropy.time import Time
from astropy.coordinates import Angle, SkyCoord
import astropy.constants as apc
from astropy.table import Table
import numpy as np
from py.path import local

# hickle imports
import hickle.loaders.load_astropy as load_astropy


# Set the current working directory to the temporary directory
local.get_temproot().chdir()

# %% FIXTURES

@pytest.fixture
def h5_data(request):
    """
    create dummy hdf5 test data file for testing PyContainer and H5NodeFilterProxy
    """
    dummy_file = h5.File('test_load_builtins.hdf5','w')
    dummy_file = h5.File('load_numpy_{}.hdf5'.format(request.function.__name__),'w')
    filename = dummy_file.filename
    test_data = dummy_file.create_group("root_group")
    yield test_data
    dummy_file.close()

# %% FUNCTION DEFINITIONS
def test_create_astropy_quantity(h5_data,compression_kwargs):
    """
    test proper storage and loading of astropy quantities
    """

    for index,uu in enumerate(['m^3', 'm^3 / s', 'kg/pc']):
        a = Quantity(7, unit=uu)
        h_dataset,subitems = load_astropy.create_astropy_quantity(a,h5_data,"quantity{}".format(index),**compression_kwargs)
        assert isinstance(h_dataset,h5.Dataset) and not subitems and iter(subitems)
        a_unit_string = a.unit.to_string()
        assert h_dataset.attrs['unit'] in ( a_unit_string.encode("ascii"),a_unit_string) and h_dataset[()] == a.value
        reloaded = load_astropy.load_astropy_quantity_dataset(h_dataset,b'astropy_quantity',Quantity)
        assert reloaded == a and reloaded.unit == a.unit
        a *= a
        h_dataset,subitems = load_astropy.create_astropy_quantity(a,h5_data,"quantity_sqr{}".format(index),**compression_kwargs)
        assert isinstance(h_dataset,h5.Dataset) and not subitems and iter(subitems)
        a_unit_string = a.unit.to_string()
        assert h_dataset.attrs['unit'] in ( a_unit_string.encode("ascii"),a_unit_string) and h_dataset[()] == a.value
        reloaded = load_astropy.load_astropy_quantity_dataset(h_dataset,b'astropy_quantity',Quantity)
        assert reloaded == a and reloaded.unit == a.unit


def test_create_astropy_constant(h5_data,compression_kwargs):

    """
    test proper storage and loading of astropy constants
    """

    h_dataset,subitems = load_astropy.create_astropy_constant(apc.G,h5_data,"apc_G",**compression_kwargs)
    assert isinstance(h_dataset,h5.Dataset) and not subitems and iter(subitems)
    apc_G_unit_string = apc.G.unit.to_string()
    assert h_dataset.attrs["unit"] in (apc_G_unit_string.encode('ascii'),apc_G_unit_string)
    assert h_dataset.attrs["abbrev"] in (apc.G.abbrev.encode('ascii'),apc.G.abbrev)
    assert h_dataset.attrs["name"] in ( apc.G.name.encode('ascii'),apc.G.name)
    assert h_dataset.attrs["reference"] in ( apc.G.reference.encode('ascii'),apc.G.reference)
    assert h_dataset.attrs["uncertainty"] == apc.G.uncertainty
    reloaded = load_astropy.load_astropy_constant_dataset(h_dataset,b'astropy_constant',apc.G.__class__)
    assert reloaded == apc.G and reloaded.dtype == apc.G.dtype

    h_dataset,subitems = load_astropy.create_astropy_constant(apc.cgs.e,h5_data,"apc_cgs_e",**compression_kwargs)
    assert isinstance(h_dataset,h5.Dataset) and not subitems and iter(subitems)
    assert h_dataset.attrs["unit"] in ( apc.cgs.e.unit.to_string().encode('ascii'),apc.cgs.e.unit)
    assert h_dataset.attrs["abbrev"] in ( apc.cgs.e.abbrev.encode('ascii'),apc.cgs.e.abbrev)
    assert h_dataset.attrs["name"] in (apc.cgs.e.name.encode('ascii'),apc.cgs.e.name)
    assert h_dataset.attrs["reference"] in ( apc.cgs.e.reference.encode('ascii'),apc.cgs.e.reference)
    assert h_dataset.attrs["uncertainty"]  == apc.cgs.e.uncertainty
    assert h_dataset.attrs["system"] in ( apc.cgs.e.system.encode('ascii'),apc.cgs.e.system )
    reloaded = load_astropy.load_astropy_constant_dataset(h_dataset,b'astropy_constant',apc.cgs.e.__class__)
    assert reloaded == apc.cgs.e and reloaded.dtype == apc.cgs.e.dtype


def test_astropy_table(h5_data,compression_kwargs):
    """
    test proper storage and loading of astropy table
    """
    t = Table([[1, 2], [3, 4]], names=('a', 'b'), meta={'name': 'test_thing'})

    h_dataset,subitems = load_astropy.create_astropy_table(t,h5_data,"astropy_table",**compression_kwargs)
    assert isinstance(h_dataset,h5.Dataset) and not subitems and iter(subitems)
    assert (
        np.all(h_dataset.attrs['colnames'] == [ cname.encode('ascii') for cname in t.colnames]) or
        np.all(h_dataset.attrs['colnames'] == [ cname for cname in t.colnames])
    )
        
    for metakey,metavalue in t.meta.items():
        assert h_dataset.attrs[metakey] == metavalue
    assert h_dataset.dtype == t.as_array().dtype
    reloaded = load_astropy.load_astropy_table(h_dataset,b'astropy_table',t.__class__)
    assert reloaded.meta == t.meta and reloaded.dtype == t.dtype
    assert np.allclose(t['a'].astype('float32'),reloaded['a'].astype('float32'))
    assert np.allclose(t['b'].astype('float32'),reloaded['b'].astype('float32'))


def test_astropy_quantity_array(h5_data,compression_kwargs):
    """
    test proper storage and loading of array of astropy quantities 
    """
    a = Quantity([1, 2, 3], unit='m')
    h_dataset,subitems = load_astropy.create_astropy_quantity(a,h5_data,"quantity_array",**compression_kwargs)
    assert isinstance(h_dataset,h5.Dataset) and not subitems and iter(subitems)
    assert h_dataset.attrs['unit'] in (a.unit.to_string().encode("ascii"),a.unit.to_string()) and np.all(h_dataset[()] == a.value)
    reloaded = load_astropy.load_astropy_quantity_dataset(h_dataset,b'astropy_quantity',Quantity)
    assert np.all(reloaded == a) and reloaded.unit == a.unit


def test_astropy_time_array(h5_data,compression_kwargs):
    """
    test proper storage and loading of astropy time representations
    """

    times = ['1999-01-01T00:00:00.123456789', '2010-01-01T00:00:00']
    t1 = Time(times, format='isot', scale='utc')

    h_dataset,subitems = load_astropy.create_astropy_time(t1,h5_data,'time1',**compression_kwargs)
    assert isinstance(h_dataset,h5.Dataset) and not subitems and iter(subitems)
    assert h_dataset.attrs['format'] in (str(t1.format).encode('ascii'),str(t1.format))
    assert h_dataset.attrs['scale'] in (str(t1.scale).encode('ascii'),str(t1.scale))
    assert h_dataset.attrs['np_dtype'] in ( t1.value.dtype.str.encode('ascii'),t1.value.dtype.str)
    reloaded = load_astropy.load_astropy_time_dataset(h_dataset,b'astropy_time',t1.__class__)
    assert reloaded.value.shape == t1.value.shape
    assert reloaded.format == t1.format
    assert reloaded.scale == t1.scale
    for index in range(len(t1)):
        assert reloaded.value[index] == t1.value[index]
    del h_dataset.attrs['np_dtype']

    reloaded = load_astropy.load_astropy_time_dataset(h_dataset,b'astropy_time',t1.__class__)
    assert reloaded.value.shape == t1.value.shape
    assert reloaded.format == t1.format
    assert reloaded.scale == t1.scale
    for index in range(len(t1)):
        assert reloaded.value[index] == t1.value[index]

    times = [58264, 58265, 58266]
    t1 = Time(times, format='mjd', scale='utc')
    h_dataset,subitems = load_astropy.create_astropy_time(t1,h5_data,'time2',**compression_kwargs)
    assert isinstance(h_dataset,h5.Dataset) and not subitems and iter(subitems)
    assert h_dataset.attrs['format'] in( str(t1.format).encode('ascii'),str(t1.format))
    assert h_dataset.attrs['scale'] in ( str(t1.scale).encode('ascii'),str(t1.scale))
    assert h_dataset.attrs['np_dtype'] in( t1.value.dtype.str.encode('ascii'),t1.value.dtype.str)
    reloaded = load_astropy.load_astropy_time_dataset(h_dataset,b'astropy_time',t1.__class__)
    assert reloaded.value.shape == t1.value.shape
    assert reloaded.format == t1.format
    assert reloaded.scale == t1.scale
    for index in range(len(t1)):
        assert reloaded.value[index] == t1.value[index]


def test_astropy_angle(h5_data,compression_kwargs):
    """
    test proper storage of astropy angles
    """

    for index,uu in enumerate(['radian', 'degree']):
        a = Angle(1.02, unit=uu)
        h_dataset,subitems = load_astropy.create_astropy_angle(a,h5_data,"angle_{}".format(uu),**compression_kwargs)
        assert isinstance(h_dataset,h5.Dataset) and not subitems and iter(subitems)
        assert h_dataset.attrs['unit'] in( a.unit.to_string().encode('ascii'),a.unit.to_string())
        assert h_dataset[()] == a.value
        reloaded = load_astropy.load_astropy_angle_dataset(h_dataset,b'astropy_angle',a.__class__)
        assert reloaded == a and reloaded.unit == a.unit


def test_astropy_angle_array(h5_data,compression_kwargs):
    """
    test proper storage and loading of arrays of astropy angles
    """
    a = Angle([1, 2, 3], unit='degree')
    h_dataset,subitems = load_astropy.create_astropy_angle(a,h5_data,"angle_array",**compression_kwargs)
    assert isinstance(h_dataset,h5.Dataset) and not subitems and iter(subitems)
    assert h_dataset.attrs['unit'] in (a.unit.to_string().encode('ascii'),a.unit.to_string())
    assert np.allclose(h_dataset[()] , a.value )
    reloaded = load_astropy.load_astropy_angle_dataset(h_dataset,b'astropy_angle',a.__class__)
    assert np.all(reloaded ==  a) and reloaded.unit == a.unit

def test_astropy_skycoord(h5_data,compression_kwargs):
    """
    test proper storage and loading of astropy sky coordinates
    """

    ra = Angle('1d20m', unit='degree')
    dec = Angle('33d0m0s', unit='degree')
    radec = SkyCoord(ra, dec)
    h_dataset,subitems = load_astropy.create_astropy_skycoord(radec,h5_data,"astropy_skycoord_1",**compression_kwargs)
    assert isinstance(h_dataset,h5.Dataset) and not subitems and iter(subitems)
    assert h_dataset[()][...,0] == radec.data.lon.value
    assert h_dataset[()][...,1] == radec.data.lat.value
    assert h_dataset.attrs['lon_unit'] in ( radec.data.lon.unit.to_string().encode('ascii'),radec.data.lon.unit.to_string())
    assert h_dataset.attrs['lat_unit'] in ( radec.data.lat.unit.to_string().encode('ascii'),radec.data.lat.unit.to_string())
    reloaded = load_astropy.load_astropy_skycoord_dataset(h_dataset,b'astropy_skycoord',radec.__class__)
    assert np.allclose(reloaded.ra.value,radec.ra.value)
    assert np.allclose(reloaded.dec.value,radec.dec.value)

    ra = Angle('1d20m', unit='hourangle')
    dec = Angle('33d0m0s', unit='degree')
    radec = SkyCoord(ra, dec)
    h_dataset,subitems = load_astropy.create_astropy_skycoord(radec,h5_data,"astropy_skycoord_2",**compression_kwargs)
    assert isinstance(h_dataset,h5.Dataset) and not subitems and iter(subitems)
    assert h_dataset[()][...,0] == radec.data.lon.value
    assert h_dataset[()][...,1] == radec.data.lat.value
    assert h_dataset.attrs['lon_unit'] in (radec.data.lon.unit.to_string().encode('ascii'),radec.data.lon.unit.to_string())
    assert h_dataset.attrs['lat_unit'] in ( radec.data.lat.unit.to_string().encode('ascii'),radec.data.lat.unit.to_string())
    reloaded = load_astropy.load_astropy_skycoord_dataset(h_dataset,b'astropy_skycoord',radec.__class__)
    assert reloaded.ra.value == radec.ra.value
    assert reloaded.dec.value == radec.dec.value

def test_astropy_skycoord_array(h5_data,compression_kwargs):
    """
    test proper storage and loading of astropy sky coordinates
    """

    ra = Angle(['1d20m', '0d21m'], unit='degree')
    dec = Angle(['33d0m0s', '-33d01m'], unit='degree')
    radec = SkyCoord(ra, dec)
    h_dataset,subitems = load_astropy.create_astropy_skycoord(radec,h5_data,"astropy_skycoord_1",**compression_kwargs)
    assert isinstance(h_dataset,h5.Dataset) and not subitems and iter(subitems)
    assert np.allclose(h_dataset[()][...,0],radec.data.lon.value)
    assert np.allclose(h_dataset[()][...,1],radec.data.lat.value)
    assert h_dataset.attrs['lon_unit'] in ( radec.data.lon.unit.to_string().encode('ascii'),radec.data.lon.unit.to_string())
    assert h_dataset.attrs['lat_unit'] in ( radec.data.lat.unit.to_string().encode('ascii'),radec.data.lat.unit.to_string())
    reloaded = load_astropy.load_astropy_skycoord_dataset(h_dataset,b'astropy_skycoord',radec.__class__)
    assert np.allclose(reloaded.ra.value,radec.ra.value)
    assert np.allclose(reloaded.dec.value,radec.dec.value)

    ra = Angle([['1d20m', '0d21m'], ['1d20m', '0d21m']], unit='hourangle')
    dec = Angle([['33d0m0s', '33d01m'], ['33d0m0s', '33d01m']], unit='degree')
    radec = SkyCoord(ra, dec)
    h_dataset,subitems = load_astropy.create_astropy_skycoord(radec,h5_data,"astropy_skycoord_2",**compression_kwargs)
    assert isinstance(h_dataset,h5.Dataset) and not subitems and iter(subitems)
    assert np.allclose(h_dataset[()][...,0],radec.data.lon.value)
    assert np.allclose(h_dataset[()][...,1],radec.data.lat.value)
    assert h_dataset.attrs['lon_unit'] in ( radec.data.lon.unit.to_string().encode('ascii'),radec.data.lon.unit.to_string())
    assert h_dataset.attrs['lat_unit'] in ( radec.data.lat.unit.to_string().encode('ascii'),radec.data.lat.unit.to_string())
    reloaded = load_astropy.load_astropy_skycoord_dataset(h_dataset,b'astropy_skycoord',radec.__class__)
    assert np.allclose(reloaded.ra.value,radec.ra.value)
    assert np.allclose(reloaded.dec.value,radec.dec.value)
    assert reloaded.ra.shape == radec.ra.shape
    assert reloaded.dec.shape == radec.dec.shape

# %% MAIN SCRIPT
if __name__ == "__main__":
    from _pytest.fixtures import FixtureRequest
    from conftest import compression_kwargs
    for h5_root,keywords in (
        ( h5_data(request),compression_kwargs(request) )
        for request in (FixtureRequest(test_create_astropy_quantity),)
    ):
        test_create_astropy_quantity(h5_root,keywords)
    for h5_root,keywords in (
        ( h5_data(request),compression_kwargs(request) )
        for request in (FixtureRequest(test_create_astropy_constant),)
    ):
        test_create_astropy_constant(h5_root,keywords)
    for h5_root,keywords in (
        ( h5_data(request),compression_kwargs(request) )
        for request in (FixtureRequest(test_astropy_table),)
    ):
        test_astropy_table(h5_root,keywords)
    for h5_root,keywords in (
        ( h5_data(request),compression_kwargs(request) )
        for request in (FixtureRequest(test_astropy_quantity_array),)
    ):
        test_astropy_quantity_array(h5_root,keywords)
    for h5_root,keywords in (
        ( h5_data(request),compression_kwargs(request) )
        for request in (FixtureRequest(test_astropy_time_array),)
    ):
        test_astropy_time_array(h5_root,keywords)
    for h5_root,keywords in (
        ( h5_data(request),compression_kwargs(request) )
        for request in (FixtureRequest(test_astropy_angle),)
    ):
        test_astropy_angle(h5_root,keywords)
    for h5_root,keywords in (
        ( h5_data(request),compression_kwargs(request) )
        for request in (FixtureRequest(test_astropy_angle_array),)
    ):
        test_astropy_angle_array(h5_root,keywords)
    for h5_root,keywords in (
        ( h5_data(request),compression_kwargs(request) )
        for request in (FixtureRequest(test_astropy_skycoord),)
    ):
        test_astropy_skycoord(h5_root,keywords)
    for h5_root,keywords in (
        ( h5_data(request),compression_kwargs(request) )
        for request in (FixtureRequest(test_astropy_skycoord_array),)
    ):
        test_astropy_skycoord_array(h5_root,keywords)
