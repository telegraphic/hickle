import astropy as ap
from astropy.units import Quantity
from astropy.constants import Constant, EMConstant
from astropy.table import Table

from hickle.helpers import get_type_and_data


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
    d.attrs["type"] = ['astropy_quantity']
    d.attrs['unit'] = [str(py_obj.unit)]


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
    d.attrs["type"]   = ['astropy_constant']
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
    d.attrs['type']  = ['astropy_table']
    d.attrs['colnames'] = py_obj.colnames
    for key, value in py_obj.meta.items():
     d.attrs[key] = value


def load_astropy_quantity_dataset(h_node):
    py_type, data = get_type_and_data(h_node)
    unit = h_node.attrs["unit"][0]
    q = Quantity(data, unit)
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

    t = Table(data, names=h_node.attrs["colnames"], meta=metadata)
    return t

def check_is_astropy_table(py_obj):
    return isinstance(py_obj, Table)

#######################
## Lookup dictionary ##
#######################

types_dict = {
    Quantity: create_astropy_quantity,
    Constant: create_astropy_constant,
    Table: create_astropy_table
}

hkl_types_dict = {
    'astropy_quantity' : load_astropy_quantity_dataset,
    'astropy_constant' : load_astropy_constant_dataset,
    'astropy_table'    : load_astropy_table
}

