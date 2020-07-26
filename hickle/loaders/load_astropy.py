# %% IMPORTS
# Package imports
from astropy.coordinates import Angle, SkyCoord
from astropy.constants import Constant
from astropy.table import Table
from astropy.time import Time
from astropy.units import Quantity
import numpy as np

# hickle imports


# %% FUNCTION DEFINITIONS
def create_astropy_quantity(py_obj, h_group, name, **kwargs):
    """ dumps an astropy quantity

    Args:
        py_obj: python object to dump; should be a python type (int, float,
            bool etc)
        h_group (h5.File.group): group to dump data into.
        name (str): the name of the resulting dataset

    Returns:
        dataset representing astropy quantity and empty subitems
    """

    d = h_group.create_dataset(name, data=py_obj.value, dtype='float64',
                               **kwargs)
    d.attrs['unit'] = py_obj.unit.to_string().encode('ascii')
    return d,()


def create_astropy_angle(py_obj, h_group, name, **kwargs):
    """ dumps an astropy quantity

    Args:
        py_obj: python object to dump; should be a python type (int, float,
            bool etc)
        h_group (h5.File.group): group to dump data into.
        name (str): the name of the resulting dataset

    Returns:
        dataset representing astropy angle and empty subitems
    """

    d = h_group.create_dataset(name, data=py_obj.value, dtype='float64',
                               **kwargs)
    d.attrs['unit'] = py_obj.unit.to_string().encode('ascii')
    return d,()


def create_astropy_skycoord(py_obj, h_group, name, **kwargs):
    """ dumps an astropy quantity

    Args:
        py_obj: python object to dump; should be a python type (int, float,
            bool etc)
        h_group (h5.File.group): group to dump data into.
        name (str): the name of the resulting dataset

    Returns:
        dataset representing astorpy SkyCoord and empty subitems
    """

    lon = py_obj.data.lon.value
    lat = py_obj.data.lat.value
    dd = np.stack((lon, lat), axis=-1)

    d = h_group.create_dataset(name, data=dd, dtype='float64', **kwargs)
    lon_unit = py_obj.data.lon.unit.to_string().encode('ascii')
    lat_unit = py_obj.data.lat.unit.to_string().encode('ascii')
    d.attrs['lon_unit'] = lon_unit
    d.attrs['lat_unit'] = lat_unit
    return d,()


def create_astropy_time(py_obj, h_group, name, **kwargs):
    """ dumps an astropy Time object

    Args:
        py_obj: python object to dump; should be a python type (int, float,
            bool etc)
        h_group (h5.File.group): group to dump data into.
        name (str): the name of the resulting dataset

    Returns:
        dataset representing string astropy time and empty subitems
    """

    # Need to catch string times
    if 'str' in py_obj.value.dtype.name:
        d = h_group.create_dataset(
            name,
            data = [item.encode('ascii') for item in py_obj.value ],
            **kwargs
        )
    else:
        d = h_group.create_dataset(name,data = py_obj.value,dtype = py_obj.value.dtype)
    d.attrs['np_dtype'] = py_obj.value.dtype.str.encode('ascii')

    d.attrs['format'] = str(py_obj.format).encode('ascii')
    d.attrs['scale'] = str(py_obj.scale).encode('ascii')

    return d,()


def create_astropy_constant(py_obj, h_group, name, **kwargs):
    """ dumps an astropy constant

    Args:
        py_obj: python object to dump; should be a python type (int, float,
            bool etc)
        h_group (h5.File.group): group to dump data into.
        name (str): the name of the resulting dataset

    Returns:
        dataset representing astropy constant and empty subitems list
    """

    d = h_group.create_dataset(name, data=py_obj.value, dtype='float64',
                               **kwargs)
    d.attrs["unit"] = py_obj.unit.to_string().encode('ascii')
    d.attrs["abbrev"] = py_obj.abbrev.encode('ascii')
    d.attrs["name"] = py_obj.name.encode('ascii')
    d.attrs["reference"] = py_obj.reference.encode('ascii')
    d.attrs["uncertainty"] = py_obj.uncertainty

    if py_obj.system:
        d.attrs["system"] = py_obj.system.encode('ascii')
    return d,()


def create_astropy_table(py_obj, h_group, name, **kwargs):
    """ Dump an astropy Table

    Args:
        py_obj: python object to dump; should be a python type (int, float,
            bool etc)
        h_group (h5.File.group): group to dump data into.
        name (str): the name of the resulting dataset
  
    Returns:
        dataset representing astorpy table and empty subitems list
    """
    data = py_obj.as_array()
    d = h_group.create_dataset(name, data=data, dtype=data.dtype, **kwargs)

    colnames = [cn.encode('ascii') for cn in py_obj.colnames]
    d.attrs['colnames'] = colnames
    for key, value in py_obj.meta.items():
        d.attrs[key] = value
    return d,()


def load_astropy_quantity_dataset(h_node,base_type,py_obj_type):
    """
    loads astropy unit
    """
    unit = h_node.attrs["unit"].decode('ascii')
    return py_obj_type(h_node[()], unit)


def load_astropy_time_dataset(h_node,base_type,py_obj_type):
    """
    loads astropy time
    """
    fmt = h_node.attrs["format"].decode('ascii')
    scale = h_node.attrs["scale"].decode('ascii')
    dtype = h_node.attrs.get('np_dtype','')
    if dtype:
        dtype = np.dtype(dtype)
        if 'str' in dtype.name:
            return py_obj_type(np.array([item.decode('ascii') for item in h_node[()]],dtype=dtype), format=fmt, scale=scale)
        return py_obj_type(np.array(h_node[()],dtype=dtype), format=fmt, scale=scale)
    return py_obj_type(np.array(h_node[()],dtype = h_node.dtype), format=fmt, scale=scale)


def load_astropy_angle_dataset(h_node,base_type,py_obj_type):
    """
    loads astropy angle
    """
    unit = h_node.attrs["unit"]
    q = py_obj_type(h_node[()], unit)
    return q


def load_astropy_skycoord_dataset(h_node,base_type,py_obj_type):
    """
    loads astropy SkyCoord
    """
    data = h_node[()]
    lon_unit = h_node.attrs["lon_unit"].decode('ascii')
    lat_unit = h_node.attrs["lat_unit"].decode('ascii')
    q = py_obj_type(data[..., 0], data[..., 1], unit=(lon_unit, lat_unit))
    return q


def load_astropy_constant_dataset(h_node,base_type,py_obj_type):
    """
    loads astropy constant
    """
    unit = h_node.attrs["unit"]
    abbrev = h_node.attrs["abbrev"]
    name = h_node.attrs["name"]
    ref = h_node.attrs["reference"]
    unc = h_node.attrs["uncertainty"]

    system = None
    if "system" in h_node.attrs.keys():
        system = h_node.attrs["system"]

    c = py_obj_type(abbrev, name, h_node[()], unit, unc, ref, system)
    return c


def load_astropy_table(h_node,base_type,py_obj_type):
    """
    loads astropy table
    """

    colnames = [cn.decode('ascii') for cn in h_node.attrs["colnames"]]

    t = py_obj_type(
        h_node[()],
        names = colnames,
        meta = {
            metakey:metavalue 
            for metakey,metavalue in h_node.attrs.items()
            if metakey not in {'type','base_type','colnames'}
        }
    )
    return t

# %% REGISTERS
class_register = [
    [Quantity, b'astropy_quantity', create_astropy_quantity,
     load_astropy_quantity_dataset],#, check_is_astropy_quantity_array],
    [Time, b'astropy_time', create_astropy_time, load_astropy_time_dataset],
    # check_is_astropy_quantity_array],
    [Angle, b'astropy_angle', create_astropy_angle, load_astropy_angle_dataset],
    # check_is_astropy_quantity_array],
    [SkyCoord, b'astropy_skycoord', create_astropy_skycoord,
     load_astropy_skycoord_dataset],# check_is_astropy_quantity_array],
    [Constant, b'astropy_constant', create_astropy_constant,
     load_astropy_constant_dataset],
    [Table, b'astropy_table',  create_astropy_table, load_astropy_table]
    # check_is_astropy_table]]
]

exclude_register = []
