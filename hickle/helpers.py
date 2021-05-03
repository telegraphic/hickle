# encoding: utf-8
"""
#helpers.py

Contains functions, classes and constants to be used
by all components of hickle including the loader modules
"""

# %% IMPORTS
# Built-in imports
import collections
import h5py as h5
import functools as ft

# Package imports


# %% EXCEPTION DEFINITIONS

nobody_is_my_name = ()

class NotHicklable(Exception):
    """
    object can not be mapped to proper hickle HDF5 file structure and
    thus shall be converted to pickle string before storing.
    """

class ToDoError(Exception):     # pragma: no cover
    """ An exception raised for non-implemented functionality"""
    def __str__(self):
        return "Error: this functionality hasn't been implemented yet."

# %% CLASS DEFINITIONS

class PyContainer():
    """
    Abstract base class for all PyContainer classes acting as proxy between
    h5py.Group and python object represented by the content of the h5py.Group.
    Any container type object as well as complex objects are represented
    in a tree like structure in the HDF5. PyContainer type objects ensure to
    properly map these structure when converting it into the corresponding
    python object structure.

    Parameters
    ----------
    h5_attrs (h5py.AttributeManager):
        attributes defined on h5py.Group object represented by this PyContainer

    base_type (bytes):
        the basic type used for representation in the HDF5 file

    object_type:
        type of Python object to be restored. May be used by PyContainer.convert
        to convert loaded Python object into final one.
        
    Attributes
    ----------
    base_type (bytes):
        the basic type used for representation on the HDF5 file

    object_type:
        type of Python object to be restored. Dependent upon container may
        be used by PyContainer.convert to convert loaded Python object into
        final one.
        
    """

    __slots__ = ("base_type", "object_type", "_h5_attrs", "_content","__dict__" )

    def __init__(self, h5_attrs, base_type, object_type, _content = None):
        """
        Parameters (protected):
        -----------------------
        _content (default: list):
            container to be used to collect the Python objects representing
            the sub items or the state of the final Python object. Shall only
            be set by derived PyContainer classes and not be set when default
            list container shall be used.

        """
        # the base type used to select this PyContainer
        self.base_type = base_type
        # class of python object represented by this PyContainer
        self.object_type = object_type
        # the h5_attrs structure of the h5_group to load the object_type from
        # can be used by the append and convert methods to obtain more
        # information about the container like object to be restored
        self._h5_attrs = h5_attrs
        # intermediate list, tuple, dict, etc. used to collect and store the sub items
        # when calling the append method
        self._content = _content if _content is not None else []

    def filter(self, h_parent):
        """
        PyContainer type child classes may overload this generator function
        to filter and preprocess the content of h_parent h5py.Group content 
        to ensure it can be properly processed by recursive calls to
        hickle._load function.

        Per default yields from h_parent.items(). 

        For examples see: 
            hickle.lookup.ExpandReferenceContainer.filter
            hickle.loaders.load_scipy.SparseMatrixContainer.filter
        """
        yield from h_parent.items()
 
    def append(self, name, item, h5_attrs):
        """
        adds the passed item to the content of this container.
       
        Parameters
        ----------
        name (string):
            the name of the h5py.Dataset or h5py.Group sub item was loaded from

        item:
            the Python object of the sub item

        h5_attrs:
            attributes defined on h5py.Group or h5py.Dataset sub item
            was loaded from.
        """
        self._content.append(item)

    def convert(self):
        """
        creates the final object and populates it with the items stored in the _content
        attribute. 

        Note: Must be implemented by the derived PyContainer child classes

        Returns
        -------
        py_obj:
            The final Python object loaded from file

        
        """
        raise NotImplementedError("convert method must be implemented")


class H5NodeFilterProxy():
    """
    Proxy class which allows to temporarily modify the content of h5_node.attrs
    attribute. Original attributes of underlying h5_node are left unchanged.
    
    Parameters
    ----------
    h5_node:
        node for which attributes shall be replaced by a temporary value
    """

    __slots__ = ('_h5_node','attrs','__dict__')

    def __init__(self,h5_node):
        # the h5py.Group or h5py.Dataset the attributes should temporarily
        # be modified.
        self._h5_node = h5_node
        # the temporarily modified attributes structure
        super().__setattr__( 'attrs', collections.ChainMap({}, h5_node.attrs))

    def __getattribute__(self, name):
        # for attrs and wrapped _h5_node return local copy. Any other request
        # redirect to wrapped _h5_node
        if name in {"attrs", "_h5_node"}:
            return super(H5NodeFilterProxy,self).__getattribute__(name)
        _h5_node = super(H5NodeFilterProxy,self).__getattribute__('_h5_node')
        return getattr(_h5_node, name)
        
    def __setattr__(self, name, value):
        # if wrapped _h5_node and attrs shall be set store value on local attributes
        # otherwise pass on to wrapped _h5_node
        if name in {'_h5_node'}:
            super().__setattr__(name, value)
            return
        if name in {'attrs'}: # pragma: no cover
            raise AttributeError('attribute is read-only')
        _h5_node = super().__getattribute__('_h5_node')
        setattr(_h5_node, name, value)    

    def __getitem__(self, *args, **kwargs):
        _h5_node = super().__getattribute__('_h5_node')
        return _h5_node.__getitem__(*args, **kwargs)
    # TODO as needed add more function like __getitem__ to fully proxy h5_node
    # or consider using metaclass __getattribute__ for handling special methods


class no_compression(dict):
    """
    named dict comprehension which temporarily removes any compression or
    data filter related argument from the passed iterable. 
    """

    # list of keyword parameters to filter
    __filter_keys__ = {
        "compression", "shuffle", "compression_opts", "chunks", "fletcher32", "scaleoffset"
    }

    def __init__(self, mapping):
        super().__init__((
            (key,value)
            for key,value in ( mapping.items() if isinstance(mapping,dict) else mapping )
            if key not in no_compression.__filter_keys__
        ))
        

# %% FUNCTION DEFINITIONS

def not_dumpable( py_obj, h_group, name, **kwargs): # pragma: no cover
    """
    create_dataset method attached to loader of dummy py_object which is used to
    mimic PyContainer class for  groups in legacy hickle 4.x file. 
        
    Raises
    ------
    RuntimeError:
        in any case as this function shall never be called    
    """

    raise RuntimeError("types defined by loaders not dump able")

def convert_str_attr(attrs,name,*,encoding='utf8'):
    return attrs[name].decode(encoding)

def convert_str_list_attr(attrs,name,*,encoding='utf8'):
    return [ value.decode(encoding) for value in attrs[name]]
  

if h5.version.version_tuple[0] >= 3: # pragma: no cover
    load_str_list_attr_ascii = load_str_list_attr = h5.AttributeManager.get
    load_str_attr_ascii = load_str_list_attr = h5.AttributeManager.get

else: # pragma: no cover
    load_str_list_attr_ascii = ft.partial(convert_str_list_attr,encoding='ascii')
    load_str_list_attr = convert_str_list_attr
    load_str_attr_ascii = ft.partial(convert_str_attr,encoding='ascii')
    load_str_attr = convert_str_attr

