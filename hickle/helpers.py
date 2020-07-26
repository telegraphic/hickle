# %% IMPORTS
# Built-in imports
import re
import operator
import typing
import types
import collections
import numbers

# Package imports
import dill as pickle


# %% EXCEPTION DEFINITIONS

nobody_is_my_name = ()

class NotHicklable(Exception):
    """
    object can not be mapped to proper hickle HDF5 file structure and
    thus shall be converted to pickle string before storing.
    """
    pass

# %% CLASS DEFINITIONS

class PyContainer():
    """
    Abstract base class for all PyContainer classes acting as proxy between
    h5py.Group and python object represented by the content of the h5py.Group.
    Any container type object as well as complex objects are represented
    in a tree like structure on HDF5 file which PyContainer objects ensure to
    be properly mapped before beeing converted into the final object.

    Parameters:
    -----------
        h5_attrs (h5py.AttributeManager):
            attributes defined on h5py.Group object represented by this PyContainer

        base_type (bytes):
            the basic type used for representation on the HDF5 file

        object_type:
            type of Python object to be restored. Dependent upon container may
            be used by PyContainer.convert to convert loaded Python object into
            final one.
        
    Attributes:
    -----------
        base_type (bytes):
            the basic type used for representation on the HDF5 file

        object_type:
            type of Python object to be restored. Dependent upon container may
            be used by PyContainer.convert to convert loaded Python object into
            final one.
        
    """

    __slots__ = ("base_type", "object_type", "_h5_attrs", "_content","__dict__" )

    def __init__(self,h5_attrs, base_type, object_type,_content = None):
        """
        Parameters (protected):
        -----------------------
            _content (default: list):
                container to be used to collect the Python objects representing
                the sub items or the state of the final Python object. Shall only
                be set by derived PyContainer classes and not be set by

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

    def filter(self,items):
        yield from items
 
    def append(self,name,item,h5_attrs):
        """
        adds the passed item (object) to the content of this container.
       
        Parameters:
        -----------
            name (string):
                the name of the h5py.Dataset or h5py.Group subitem was loaded from

            item:
                the Python object of the subitem

            h5_attrs:
                attributes defined on h5py.Group or h5py.Dataset object sub item
                was loaded from.
        """
        self._content.append(item)

    def convert(self):
        """
        creates the final object and populates it with the items stored in the _content slot
        must be implemented by the derived Container classes

        Returns:
        --------
            py_obj: The final Python object loaded from file

        
        """
        raise NotImplementedError("convert method must be implemented")


class H5NodeFilterProxy():
    """
    Proxy class which allows to temporarily modify h5_node.attrs content.
    Original attributes of underlying h5_node are left unchanged.
    
    Parameters:
    -----------
        h5_node:
            node for which attributes shall be replaced by a temporary value
        
    """

    __slots__ = ('_h5_node','attrs','__dict__')

    def __init__(self,h5_node):
        self._h5_node = h5_node
        self.attrs = collections.ChainMap({},h5_node.attrs)

    def __getattribute__(self,name):
        # for attrs and wrapped _h5_node return local copy any other request
        # redirect to wrapped _h5_node
        if name in {"attrs","_h5_node"}:
            return super(H5NodeFilterProxy,self).__getattribute__(name)
        _h5_node = super(H5NodeFilterProxy,self).__getattribute__('_h5_node')
        return getattr(_h5_node,name)
        
    def __setattr__(self,name,value):
        # if wrapped _h5_node and attrs shall be set store value on local attributes
        # otherwise pass on to wrapped _h5_node
        if name in {'_h5_node','attrs'}:
            super(H5NodeFilterProxy,self).__setattr__(name,value)
            return
        _h5_node = super(H5NodeFilterProxy,self).__getattribute__('_h5_node')
        setattr(_h5_node,name,value)    

    def __getitem__(self,*args,**kwargs):
        _h5_node = super(H5NodeFilterProxy,self).__getattribute__('_h5_node')
        return _h5_node.__getitem__(*args,**kwargs)
    # TODO as needed add more function like __getitem__ to fully proxy h5_node
    # or consider using metaclass __getattribute__ for handling special methods


class no_compression(dict):
    """
    subclass of dict which which temporarily removes any compression or data filter related
    arguments from the passed iterable. 
    """
    def __init__(self,mapping,**kwargs):
        super().__init__((
            (key,value)
            for key,value in ( mapping.items() if isinstance(mapping,dict) else mapping )
            if key not in {"compression","shuffle","compression_opts","chunks","fletcher32","scaleoffset"}
        ))
        

# %% FUNCTION DEFINITIONS

def not_dumpable( py_obj, h_group, name, **kwargs): # pragma: nocover
    """
    create_dataset method attached to dummy py_objects used to mimic container
    groups by older versions of hickle lacking generic PyContainer mapping
    h5py.Groups to corresponding py_object

        
    Raises:
    -------
        RuntimeError:
            in any case as this function shall never be called    
    """

    raise RuntimeError("types defined by loaders not dumpable")


