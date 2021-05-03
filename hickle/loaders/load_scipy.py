# %% IMPORTS
# Package imports
import dill as pickle
import numpy as np
import scipy
import copy
from scipy import sparse

# hickle imports
from hickle.helpers import PyContainer,H5NodeFilterProxy
from hickle.loaders.load_numpy import load_ndarray_dataset,create_np_array_dataset


# %% FUNCTION DEFINITIONS
def return_first(x):
    """
    Dummy function used as place holder type in loading legacy hickle 4.x files
    """
    raise TypeError("'return_first' not callable and deprecated. Create and use PyContainer instead.")


def create_sparse_dataset(py_obj, h_group, name, **kwargs):
    """ dumps an sparse array to h5py file

    Parameters
    ----------
    py_obj (scipy.sparse.csr_matrix,scipy.sparse.csc_matrix, scipy.sparse.bsr_matrix):
        python object to dump

    h_group (h5.File.group):
        group to dump data into.

    name (str):
        the name of the resulting dataset

    kwargs (dict):
        keyword arguments to be passed to create_dataset function

    Returns:
        Group and list of subitems to dump into
    """
    h_sparsegroup = h_group.create_group(name)
    return h_sparsegroup,(
        ('data',py_obj.data,{},kwargs),
        ('indices',py_obj.indices,{},kwargs),
        ('indptr',py_obj.indptr,{},kwargs),
        ('shape',py_obj.shape,{},kwargs)
    )

ndarray_type_string = pickle.dumps(np.ndarray)
tuple_type_string = pickle.dumps(tuple)

            
class SparseMatrixContainer(PyContainer):
    """
    PyContainer used to restore sparse Matrix
    """

    # instance attribute shadowing class method of same name
    # points per default to shadowed method
    __slots__ = ('filter',)
    
    _index_name_map = {
        'data':0,
        'indices':1,
        'indptr':2,
        'shape':3
    }

    def __init__(self,h5_attrs, base_type, object_type):
        super(SparseMatrixContainer,self).__init__(h5_attrs,base_type,object_type,_content = [None]*4)
        
        # in case object type is return_first (hickle 4.x file) than switch filter
        # to redirect loading of sub items to numpy ndarray type. Otherwise set to 
        # PyContainer.filter method
        if object_type is return_first:
            self.filter = self._redirect_to_ndarray
        else:
            self.filter = super(SparseMatrixContainer,self).filter

    def _redirect_to_ndarray(self,h_parent):
        """
        iterates through items and extracts effective object and basetype
        of sparse matrix from data subitem and remaps all subitems to 
        ndarray type exempt shape which is remapped to tuple
        """

        for name,item in h_parent.items():
            item = H5NodeFilterProxy(item)
            if name == 'data':
                self.object_type = pickle.loads(item.attrs['type'])
                self.base_type = item.attrs['base_type']
                np_dtype = item.attrs.get('np_dtype',None)
                if np_dtype is None:
                    item.attrs['np_dtype'] = item.dtype.str.encode('ascii')
            elif name not in self._index_name_map.keys():
                continue # ignore name
            if name == "shape":
                item.attrs['type'] = np.array(tuple_type_string)
                item.attrs['base_type'] = b'tuple'
            else:
                item.attrs['type'] = np.array(ndarray_type_string)
                item.attrs['base_type'] = b'ndarray'
                np_dtype = item.attrs.get('np_dtype',None)
                if np_dtype is None:
                    item.attrs['np_dtype'] = item.dtype.str.encode('ascii')
            yield name,item

    def append(self,name,item,h5_attrs):
        index = self._index_name_map.get(name,None)
        self._content[index] = item

    def convert(self):
        return self.object_type(tuple(self._content[:3]),dtype=self._content[0].dtype,shape=self._content[3])

# %% REGISTERS
class_register = [
    [scipy.sparse.csr_matrix, b'csr_matrix', create_sparse_dataset, None, SparseMatrixContainer],
    [scipy.sparse.csc_matrix, b'csc_matrix', create_sparse_dataset, None, SparseMatrixContainer],
    [scipy.sparse.bsr_matrix, b'bsr_matrix', create_sparse_dataset, None, SparseMatrixContainer]
]

exclude_register = []

