# %% IMPORTS
# Package imports
from astropy.coordinates import Angle, SkyCoord
from astropy.constants import Constant
from astropy.table import Table
from astropy.time import Time
from astropy.units import Quantity
import numpy as np

# hickle imports
from hickle.helpers import get_type_and_data


# %% FUNCTION DEFINITIONS
def create_astropy_quantity(py_obj, h_group, name, **kwargs):
    """ dumps an astropy quantity

    Args:
        py_obj: python object to dump; should be a python type (int, float,
            bool etc)
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the
            iterable.
    """

    d = h_group.create_dataset(name, data=py_obj.value, dtype='float64')
    unit = bytes(str(py_obj.unit), 'ascii')
    d.attrs['unit'] = unit
    return(d)


def create_astropy_angle(py_obj, h_group, name, **kwargs):
    """ dumps an astropy quantity

    Args:
        py_obj: python object to dump; should be a python type (int, float,
            bool etc)
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the
            iterable.
    """

    d = h_group.create_dataset(name, data=py_obj.value, dtype='float64')
    unit = str(py_obj.unit).encode('ascii')
    d.attrs['unit'] = unit
    return(d)


def create_astropy_skycoord(py_obj, h_group, name, **kwargs):
    """ dumps an astropy quantity

    Args:
        py_obj: python object to dump; should be a python type (int, float,
            bool etc)
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the
            iterable.
    """

    lat = py_obj.data.lat.value
    lon = py_obj.data.lon.value
    dd = np.column_stack((lon, lat))

    d = h_group.create_dataset(name, data=dd, dtype='float64')
    lon_unit = str(py_obj.data.lon.unit).encode('ascii')
    lat_unit = str(py_obj.data.lat.unit).encode('ascii')
    d.attrs['lon_unit'] = lon_unit
    d.attrs['lat_unit'] = lat_unit
    return(d)


def create_astropy_time(py_obj, h_group, name, **kwargs):
    """ dumps an astropy Time object

    Args:
        py_obj: python object to dump; should be a python type (int, float,
            bool etc)
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the
            iterable.
    """

    data = py_obj.value
    dtype = str(py_obj.value.dtype)

    # Need to catch string times
    if '<U' in dtype:
        dtype = dtype.replace('<U', '|S')
        print(dtype)
        data = []
        for item in py_obj.value:
            data.append(str(item).encode('ascii'))

    d = h_group.create_dataset(name, data=data, dtype=dtype)
    fmt = str(py_obj.format).encode('ascii')
    scale = str(py_obj.scale).encode('ascii')
    d.attrs['format'] = fmt
    d.attrs['scale'] = scale

    return(d)


def create_astropy_constant(py_obj, h_group, name, **kwargs):
    """ dumps an astropy constant

    Args:
        py_obj: python object to dump; should be a python type (int, float,
            bool etc)
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the
            iterable.
    """

    d = h_group.create_dataset(name, data=py_obj.value, dtype='float64')
    d.attrs["unit"] = str(py_obj.unit)
    d.attrs["abbrev"] = str(py_obj.abbrev)
    d.attrs["name"] = str(py_obj.name)
    d.attrs["reference"] = str(py_obj.reference)
    d.attrs["uncertainty"] = py_obj.uncertainty

    if py_obj.system:
        d.attrs["system"] = py_obj.system
    return(d)


def create_astropy_table(py_obj, h_group, name, **kwargs):
    """ Dump an astropy Table

    Args:
        py_obj: python object to dump; should be a python type (int, float,
            bool etc)
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the
            iterable.
    """
    data = py_obj.as_array()
    d = h_group.create_dataset(name, data=data, dtype=data.dtype, **kwargs)

    colnames = [bytes(cn, 'ascii') for cn in py_obj.colnames]
    d.attrs['colnames'] = colnames
    for key, value in py_obj.meta.items():
        d.attrs[key] = value
    return(d)


def load_astropy_quantity_dataset(h_node):
    py_type, _, data = get_type_and_data(h_node)
    unit = h_node.attrs["unit"]
    q = py_type(data, unit, copy=False)
    return q


def load_astropy_time_dataset(h_node):
    py_type, _, data = get_type_and_data(h_node)
    fmt = h_node.attrs["format"].decode('ascii')
    scale = h_node.attrs["scale"].decode('ascii')
    q = py_type(data, format=fmt, scale=scale)
    return q


def load_astropy_angle_dataset(h_node):
    py_type, _, data = get_type_and_data(h_node)
    unit = h_node.attrs["unit"]
    q = py_type(data, unit)
    return q


def load_astropy_skycoord_dataset(h_node):
    py_type, _, data = get_type_and_data(h_node)
    lon_unit = h_node.attrs["lon_unit"]
    lat_unit = h_node.attrs["lat_unit"]
    q = py_type(data[:, 0], data[:, 1], unit=(lon_unit, lat_unit))
    return q


def load_astropy_constant_dataset(h_node):
    py_type, _, data = get_type_and_data(h_node)
    unit = h_node.attrs["unit"]
    abbrev = h_node.attrs["abbrev"]
    name = h_node.attrs["name"]
    ref = h_node.attrs["reference"]
    unc = h_node.attrs["uncertainty"]

    system = None
    if "system" in h_node.attrs.keys():
        system = h_node.attrs["system"]

    c = py_type(abbrev, name, data, unit, unc, ref, system)
    return c


def load_astropy_table(h_node):
    py_type, _, data = get_type_and_data(h_node)
    metadata = dict(h_node.attrs.items())
    metadata.pop('type')
    metadata.pop('base_type')
    metadata.pop('colnames')

    colnames = [cn.decode('ascii') for cn in h_node.attrs["colnames"]]

    t = py_type(data, names=colnames, meta=metadata)
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


# %% REGISTERS
class_register = [
    [Quantity, b'astropy_quantity', create_astropy_quantity,
     load_astropy_quantity_dataset, check_is_astropy_quantity_array],
    [Time, b'astropy_time', create_astropy_time, load_astropy_time_dataset,
     check_is_astropy_quantity_array],
    [Angle, b'astropy_angle', create_astropy_angle, load_astropy_angle_dataset,
     check_is_astropy_quantity_array],
    [SkyCoord, b'astropy_skycoord', create_astropy_skycoord,
     load_astropy_skycoord_dataset, check_is_astropy_quantity_array],
    [Constant, b'astropy_constant', create_astropy_constant,
     load_astropy_constant_dataset],
    [Table, b'astropy_table',  create_astropy_table, load_astropy_table,
     check_is_astropy_table]]

exclude_register = []
