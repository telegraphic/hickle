import numpy as np
from astropy.units import Quantity
from astropy.coordinates import Angle, SkyCoord
from astropy.constants import Constant, EMConstant
from astropy.table import Table
from astropy.time import Time

from ..helpers import get_type_and_data
import six

def create_astropy_quantity(py_obj, h_group, call_id=0, **kwargs):
    """ dumps an astropy quantity

    Args:
        py_obj: python object to dump; should be a python type (int, float, bool etc)
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the iterable.
    """
    # kwarg compression etc does not work on scalars
    d = h_group.create_dataset('data_%i' % call_id, data=py_obj.value,
                               dtype='float64')     #, **kwargs)
    d.attrs["type"] = [b'astropy_quantity']
    if six.PY3:
        unit = bytes(str(py_obj.unit), 'ascii')
    else:
        unit = str(py_obj.unit)
    d.attrs['unit'] = [unit]

def create_astropy_angle(py_obj, h_group, call_id=0, **kwargs):
    """ dumps an astropy quantity

    Args:
        py_obj: python object to dump; should be a python type (int, float, bool etc)
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the iterable.
    """
    # kwarg compression etc does not work on scalars
    d = h_group.create_dataset('data_%i' % call_id, data=py_obj.value,
                               dtype='float64')     #, **kwargs)
    d.attrs["type"] = [b'astropy_angle']
    if six.PY3:
        unit = str(py_obj.unit).encode('ascii')
    else:
        unit = str(py_obj.unit)
    d.attrs['unit'] = [unit]

def create_astropy_skycoord(py_obj, h_group, call_id=0, **kwargs):
    """ dumps an astropy quantity

    Args:
        py_obj: python object to dump; should be a python type (int, float, bool etc)
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the iterable.
    """
    # kwarg compression etc does not work on scalars
    lat = py_obj.data.lat.value
    lon = py_obj.data.lon.value
    dd = np.stack((lon, lat), axis=-1)

    d = h_group.create_dataset('data_%i' % call_id, data=dd,
                               dtype='float64')     #, **kwargs)
    d.attrs["type"] = [b'astropy_skycoord']
    if six.PY3:
        lon_unit = str(py_obj.data.lon.unit).encode('ascii')
        lat_unit = str(py_obj.data.lat.unit).encode('ascii')
    else:
        lon_unit = str(py_obj.data.lon.unit)
        lat_unit = str(py_obj.data.lat.unit)
    d.attrs['lon_unit'] = [lon_unit]
    d.attrs['lat_unit'] = [lat_unit]

def create_astropy_time(py_obj, h_group, call_id=0, **kwargs):
    """ dumps an astropy Time object

    Args:
        py_obj: python object to dump; should be a python type (int, float, bool etc)
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the iterable.
    """

    # kwarg compression etc does not work on scalars
    data = py_obj.value
    dtype = str(py_obj.value.dtype)

    # Need to catch string times
    if '<U' in dtype:
        dtype = dtype.replace('<U', '|S')
        print(dtype)
        data = []
        for item in py_obj.value:
            data.append(str(item).encode('ascii'))

    d = h_group.create_dataset('data_%i' % call_id, data=data, dtype=dtype)     #, **kwargs)
    d.attrs["type"] = [b'astropy_time']
    if six.PY2:
        fmt   = str(py_obj.format)
        scale = str(py_obj.scale)
    else:
        fmt   = str(py_obj.format).encode('ascii')
        scale = str(py_obj.scale).encode('ascii')
    d.attrs['format'] = [fmt]
    d.attrs['scale']  = [scale]

def create_astropy_constant(py_obj, h_group, call_id=0, **kwargs):
    """ dumps an astropy constant

    Args:
        py_obj: python object to dump; should be a python type (int, float, bool etc)
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the iterable.
    """
    # kwarg compression etc does not work on scalars
    d = h_group.create_dataset('data_%i' % call_id, data=py_obj.value,
                               dtype='float64')     #, **kwargs)
    d.attrs["type"]   = [b'astropy_constant']
    d.attrs["unit"]   = [str(py_obj.unit)]
    d.attrs["abbrev"] = [str(py_obj.abbrev)]
    d.attrs["name"]   = [str(py_obj.name)]
    d.attrs["reference"] = [str(py_obj.reference)]
    d.attrs["uncertainty"] = [py_obj.uncertainty]

    if py_obj.system:
        d.attrs["system"] = [py_obj.system]


def create_astropy_table(py_obj, h_group, call_id=0, **kwargs):
    """ Dump an astropy Table

    Args:
        py_obj: python object to dump; should be a python type (int, float, bool etc)
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the iterable.
    """
    data = py_obj.as_array()
    d = h_group.create_dataset('data_%i' % call_id, data=data, dtype=data.dtype, **kwargs)
    d.attrs['type']  = [b'astropy_table']

    if six.PY3:
        colnames = [bytes(cn, 'ascii') for cn in py_obj.colnames]
    else:
        colnames = py_obj.colnames
    d.attrs['colnames'] = colnames
    for key, value in py_obj.meta.items():
     d.attrs[key] = value


def load_astropy_quantity_dataset(h_node):
    py_type, data = get_type_and_data(h_node)
    unit = h_node.attrs["unit"][0]
    q = Quantity(data, unit, copy=False)
    return q

def load_astropy_time_dataset(h_node):
    py_type, data = get_type_and_data(h_node)
    if six.PY3:
        fmt = h_node.attrs["format"][0].decode('ascii')
        scale = h_node.attrs["scale"][0].decode('ascii')
    else:
        fmt = h_node.attrs["format"][0]
        scale = h_node.attrs["scale"][0]
    q = Time(data, format=fmt, scale=scale)
    return q

def load_astropy_angle_dataset(h_node):
    py_type, data = get_type_and_data(h_node)
    unit = h_node.attrs["unit"][0]
    q = Angle(data, unit)
    return q

def load_astropy_skycoord_dataset(h_node):
    py_type, data = get_type_and_data(h_node)
    lon_unit = h_node.attrs["lon_unit"][0]
    lat_unit = h_node.attrs["lat_unit"][0]
    q = SkyCoord(data[..., 0], data[..., 1], unit=(lon_unit, lat_unit))
    return q

def load_astropy_constant_dataset(h_node):
    py_type, data = get_type_and_data(h_node)
    unit   = h_node.attrs["unit"][0]
    abbrev = h_node.attrs["abbrev"][0]
    name   = h_node.attrs["name"][0]
    ref    = h_node.attrs["reference"][0]
    unc    = h_node.attrs["uncertainty"][0]

    system = None
    if "system" in h_node.attrs.keys():
        system = h_node.attrs["system"][0]

    c = Constant(abbrev, name, data, unit, unc, ref, system)
    return c

def load_astropy_table(h_node):
    py_type, data = get_type_and_data(h_node)
    metadata = dict(h_node.attrs.items())
    metadata.pop('type')
    metadata.pop('colnames')

    if six.PY3:
        colnames = [cn.decode('ascii') for cn in h_node.attrs["colnames"]]
    else:
        colnames = h_node.attrs["colnames"]

    t = Table(data, names=colnames, meta=metadata)
    return t

def check_is_astropy_table(py_obj):
    return isinstance(py_obj, Table)

def check_is_astropy_quantity_array(py_obj):
    if isinstance(py_obj, Quantity) or isinstance(py_obj, Time) or \
       isinstance(py_obj, Angle) or isinstance(py_obj, SkyCoord):
        if py_obj.isscalar:
            return False
        else:
            return True
    else:
        return False


#####################
# Lookup dictionary #
#####################

class_register = [
    [Quantity, b'astropy_quantity', create_astropy_quantity, load_astropy_quantity_dataset,
     True, check_is_astropy_quantity_array],
    [Time,     b'astropy_time', create_astropy_time, load_astropy_time_dataset,
     True, check_is_astropy_quantity_array],
    [Angle,    b'astropy_angle', create_astropy_angle, load_astropy_angle_dataset,
     True, check_is_astropy_quantity_array],
    [SkyCoord, b'astropy_skycoord', create_astropy_skycoord, load_astropy_skycoord_dataset,
     True, check_is_astropy_quantity_array],
    [Constant, b'astropy_constant', create_astropy_constant, load_astropy_constant_dataset,
     True, None],
    [Table,    b'astropy_table',  create_astropy_table, load_astropy_table,
     True, check_is_astropy_table]
]
