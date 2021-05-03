#! /usr/bin/env python
# encoding: utf-8
"""
# test_load_scipy

Unit tests for hickle module -- scipy loader.

"""
# %% IMPORTS
# Package imports
import pytest
import h5py as h5
import numpy as np
import dill as pickle
from scipy.sparse import csr_matrix, csc_matrix, bsr_matrix
from py.path import local

# %% HICKLE imports
import hickle.loaders.load_scipy as load_scipy

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

def test_return_first_function_type():
    with pytest.raises(TypeError):
        load_scipy.return_first(['anything','some other thins','nothing'])

def test_create_sparse_dataset(h5_data,compression_kwargs):
    """
    test creation and loading of sparse matrix 
    """

    # create all possible kinds of sparse matrix representations
    row = np.array([0, 0, 1, 2, 2, 2])
    col = np.array([0, 2, 2, 0, 1, 2])
    data = np.array([1, 2, 3, 4, 5, 6])
    sm1 = csr_matrix((data, (row, col)), shape=(3, 3))
    sm2 = csc_matrix((data, (row, col)), shape=(3, 3))

    indptr = np.array([0, 2, 3, 6])
    indices = np.array([0, 2, 2, 0, 1, 2])
    data = np.array([1, 2, 3, 4, 5, 6]).repeat(4).reshape([6, 2, 2])
    sm3 = bsr_matrix((data, indices, indptr), shape=(6, 6))

    # check that csr type matrix is properly stored and loaded
    h_datagroup,subitems = load_scipy.create_sparse_dataset(sm1,h5_data,"csr_matrix",**compression_kwargs)
    assert isinstance(h_datagroup,h5.Group) and iter(subitems)
    seen_items = dict((key,False) for key in ("data",'indices','indptr','shape'))
    sparse_container = load_scipy.SparseMatrixContainer(h_datagroup.attrs,b'csr_matrix',csr_matrix)
    for name,item,attrs,kwargs in subitems:
        assert not seen_items[name]
        seen_items[name] = True
        sparse_container.append(name,item,attrs)
    reloaded = sparse_container.convert()
    assert np.all(reloaded.data == sm1.data) and reloaded.dtype == sm1.dtype and reloaded.shape == sm1.shape

    # check that csc type matrix is properly stored and loaded
    h_datagroup,subitems = load_scipy.create_sparse_dataset(sm2,h5_data,"csc_matrix",**compression_kwargs)
    assert isinstance(h_datagroup,h5.Group) and iter(subitems)
    seen_items = dict((key,False) for key in ("data",'indices','indptr','shape'))
    sparse_container = load_scipy.SparseMatrixContainer(h_datagroup.attrs,b'csc_matrix',csc_matrix)
    for name,item,attrs,kwargs in subitems:
        assert not seen_items[name]
        seen_items[name] = True
        sparse_container.append(name,item,attrs)
    reloaded = sparse_container.convert()
    assert np.all(reloaded.data == sm2.data) and reloaded.dtype == sm2.dtype and reloaded.shape == sm2.shape

    # check that bsr type matrix is properly stored and loaded
    h_datagroup,subitems = load_scipy.create_sparse_dataset(sm3,h5_data,"bsr_matrix",**compression_kwargs)
    assert isinstance(h_datagroup,h5.Group) and iter(subitems)
    seen_items = dict((key,False) for key in ("data",'indices','indptr','shape'))
    sparse_container = load_scipy.SparseMatrixContainer(h_datagroup.attrs,b'bsr_matrix',bsr_matrix)
    for name,item,attrs,kwargs in subitems:
        assert not seen_items[name]
        seen_items[name] = True
        sparse_container.append(name,item,attrs)
    reloaded = sparse_container.convert()
    assert np.all(reloaded.data == sm3.data) and reloaded.dtype == sm3.dtype and reloaded.shape == sm3.shape

    # mimic hickle version 4.0.0 format to represent crs type matrix
    h_datagroup,subitems = load_scipy.create_sparse_dataset(sm1,h5_data,"csr_matrix_filtered",**compression_kwargs)
    sparse_container = load_scipy.SparseMatrixContainer(h_datagroup.attrs,b'csr_matrix',load_scipy.return_first)
    for name,item,attrs,kwargs in subitems:
        h_dataset = h_datagroup.create_dataset(name,data=item)
        if name == "data":
            attrs["type"] = np.array(pickle.dumps(sm1.__class__))
            attrs["base_type"] = b'csr_matrix'
        h_dataset.attrs.update(attrs)

    # check that dataset representing hickle 4.0.0 representation of sparse matrix
    # is properly recognized by SparseMatrixContainer.filter method and sub items of
    # sparse matrix group are properly adjusted to be safely loaded by SparseMatrixContainer
    for name,h_dataset in sparse_container.filter(h_datagroup):
        if name == "shape":
            sparse_container.append(name,tuple(h_dataset[()]),h_dataset.attrs)
        else:
            sparse_container.append(name,np.array(h_dataset[()]),h_dataset.attrs)
    reloaded = sparse_container.convert()
    assert np.all(reloaded.data == sm1.data) and reloaded.dtype == sm1.dtype and reloaded.shape == sm1.shape

    # verify that SparseMatrixContainer.filter method ignores any items which
    # are not recognized by SparseMatrixContainer update or convert method
    h_datagroup.create_dataset("ignoreme",data=12)
    for name,h_dataset in sparse_container.filter(h_datagroup):
        if name == "shape":
            sparse_container.append(name,tuple(h_dataset[()]),h_dataset.attrs)
        else:
            sparse_container.append(name,np.array(h_dataset[()]),h_dataset.attrs)
    reloaded = sparse_container.convert()
    assert np.all(reloaded.data == sm1.data) and reloaded.dtype == sm1.dtype and reloaded.shape == sm1.shape
    
    
# %% MAIN SCRIPT
if __name__ == "__main__":
    from _pytest.fixtures import FixtureRequest
    from hickle.tests.conftest import compression_kwargs

    test_return_first_function_type()
    for h5_root,keywords in (
        ( h5_data(request),compression_kwargs(request) )
        for request in (FixtureRequest(test_create_sparse_dataset),)
    ):
        test_create_sparse_dataset(h5_root,keywords)
