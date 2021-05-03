#! /usr/bin/env python
# encoding: utf-8
"""
# test_load_builtins

Unit tests for hickle module -- builtins loader.

"""

import pytest
import collections
import itertools

# %% IMPORTS
# Package imports
import h5py as h5
import numpy as np
from py.path import local

# hickle imports
import hickle.loaders.load_builtins as load_builtins
import hickle.helpers as helpers


# Set current working directory to the temporary directory
local.get_temproot().chdir()


# %% TEST DATA

dummy_data = (1,2,3)

# %% FIXTURES

@pytest.fixture
def h5_data(request):
    """
    create dummy hdf5 test data file providing parent group
    hosting created datasets and groups. Name of test function
    is included in filename
    """
    dummy_file = h5.File('load_builtins_{}.hdf5'.format(request.function.__name__),'w')
    filename = dummy_file.filename
    test_data = dummy_file.create_group("root_group")
    yield test_data
    dummy_file.close()


# %% FUNCTION DEFINITIONS

def test_scalar_dataset(h5_data,compression_kwargs):
    """
    tests creation and loading of datasets for scalar values
    """

    # check that scalar value is properly handled
    floatvalue = 5.2
    h_dataset,subitems= load_builtins.create_scalar_dataset(floatvalue,h5_data,"floatvalue",**compression_kwargs)
    assert isinstance(h_dataset,h5.Dataset) and h_dataset[()] == floatvalue
    assert not [ item for item in subitems ]
    assert load_builtins.load_scalar_dataset(h_dataset,b'float',float) == floatvalue

    # check that integer value less than 64 bit is stored as int
    intvalue = 11
    h_dataset,subitems = load_builtins.create_scalar_dataset(intvalue,h5_data,"intvalue",**compression_kwargs)
    assert isinstance(h_dataset,h5.Dataset) and h_dataset[()] == intvalue
    assert not [ item for item in subitems ]
    assert load_builtins.load_scalar_dataset(h_dataset,b'int',int) == intvalue

    # check that integer larger than 64 bit is stored as ASCII byte string
    non_mappable_int = int(2**65)
    h_dataset,subitems = load_builtins.create_scalar_dataset(non_mappable_int,h5_data,"non_mappable_int",**compression_kwargs)
    assert isinstance(h_dataset,h5.Dataset)
    assert bytearray(h_dataset[()]) == str(non_mappable_int).encode('utf8')
    assert not [ item for item in subitems ]
    assert load_builtins.load_scalar_dataset(h_dataset,b'int',int) == non_mappable_int

    # check that integer larger than 64 bit is stored as ASCII byte string
    non_mappable_neg_int = -int(-2**63-1)
    h_dataset,subitems = load_builtins.create_scalar_dataset(non_mappable_neg_int,h5_data,"non_mappable_neg_int",**compression_kwargs)
    assert isinstance(h_dataset,h5.Dataset)
    assert bytearray(h_dataset[()]) == str(non_mappable_neg_int).encode('utf8')
    assert not [ item for item in subitems ]
    assert load_builtins.load_scalar_dataset(h_dataset,b'int',int) == non_mappable_neg_int

def test_load_hickle_4_0_X_string(h5_data):
    string_data = "just test me as utf8 string"
    bytes_data = string_data.encode('utf8')
    if h5.version.version_tuple[0] >= 3:
        utf_entry = h5_data.create_dataset('utf_entry',data = string_data)#,dtype = 'U{}'.format(len(string_data)))
        bytes_entry = h5_data.create_dataset('bytes_entry',data = bytes_data,dtype = 'S{}'.format(len(bytes_data)))
    else:
        utf_entry = h5_data.create_dataset('utf_entry',data = string_data)
        bytes_entry = h5_data.create_dataset('bytes_entry',data = bytes_data)
    assert load_builtins.load_hickle_4_x_string(utf_entry,b'str',str) == string_data
    bytes_entry.attrs['str_type'] = b'str'
    assert load_builtins.load_hickle_4_x_string(bytes_entry,b'str',str) == string_data
    object_entry = h5_data.create_dataset('utf_h5py2_entry',data = string_data,dtype = np.dtype('O',metadata={'vlen':bytes}))
    assert load_builtins.load_hickle_4_x_string(object_entry,b'str',bytes) == bytes_data
    
    
def test_non_dataset(h5_data,compression_kwargs):
    """
     that None value is properly stored
    """
    h_dataset,subitems = load_builtins.create_none_dataset(None,h5_data,"None_value",**compression_kwargs)
    assert isinstance(h_dataset,h5.Dataset) and h_dataset.shape is None and h_dataset.dtype == 'V1'
    assert not [ item for item in subitems ]
    assert load_builtins.load_none_dataset(h_dataset,b'None',None.__class__) is None


def test_listlike_dataset(h5_data,compression_kwargs):
    """
    test storing and loading of list like data 
    """

    # check that empty tuple is stored properly
    empty_tuple = ()
    h_dataset,subitems = load_builtins.create_listlike_dataset(empty_tuple, h5_data, "empty_tuple",**compression_kwargs)
    assert isinstance(h_dataset,h5.Dataset) and h_dataset.size is None
    assert not subitems and iter(subitems)
    assert load_builtins.load_list_dataset(h_dataset,b'tuple',tuple) == empty_tuple

    # check that string data is stored properly stored as array of bytes
    # which supports compression
    stringdata = "string_data"
    h_dataset,subitems = load_builtins.create_listlike_dataset(stringdata, h5_data, "string_data",**compression_kwargs)
    assert isinstance(h_dataset,h5.Dataset) and not [ item for item in subitems ]
    assert bytearray(h_dataset[()]).decode("utf8") == stringdata
    assert h_dataset.attrs["str_type"] in ('str',b'str')
    assert load_builtins.load_list_dataset(h_dataset,b'str',str) == stringdata

    # check that byte string is properly stored as array of bytes which
    # supports compression
    bytesdata = b'bytes_data'
    h_dataset,subitems = load_builtins.create_listlike_dataset(bytesdata, h5_data, "bytes_data",**compression_kwargs)
    assert isinstance(h_dataset,h5.Dataset) and not [ item for item in subitems ]
    assert bytes(h_dataset[()]) == bytesdata
    assert h_dataset.attrs["str_type"] in ('bytes',b'bytes')
    assert load_builtins.load_list_dataset(h_dataset,b'bytes',bytes) == bytesdata

    # check that string dataset created by hickle 4.0.x is properly loaded 
    # utilizing numpy.array method. Mimic dumped data
    h_dataset = h5_data.create_dataset("legacy_np_array_bytes_data",data=np.array(stringdata.encode('utf8')))
    h_dataset.attrs['str_type'] = b'str'
    assert load_builtins.load_list_dataset(h_dataset,b'str',str) == stringdata

    # check that list of single type is stored as dataset of same type
    homogenous_list = [ 1, 2, 3, 4, 5, 6]
    h_dataset,subitems = load_builtins.create_listlike_dataset(homogenous_list,h5_data,"homogenous_list",**compression_kwargs)
    assert isinstance(h_dataset,h5.Dataset) and not [ item for item in subitems ]
    assert h_dataset[()].tolist() == homogenous_list and h_dataset.dtype == int
    assert load_builtins.load_list_dataset(h_dataset,b'list',list) == homogenous_list

    # check that list of different scalar types for which a least common type exists
    # is stored using a dataset 
    mixed_dtype_list = [ 1, 2.5, 3.8, 4, 5, 6]
    h_dataset,subitems = load_builtins.create_listlike_dataset(mixed_dtype_list,h5_data,"mixed_dtype_list",**compression_kwargs)
    assert isinstance(h_dataset,h5.Dataset) and not [ item for item in subitems ]
    assert h_dataset[()].tolist() == mixed_dtype_list and h_dataset.dtype == float
    assert load_builtins.load_list_dataset(h_dataset,b'list',list) == mixed_dtype_list

    # check that list containing non scalar objects is converted into group
    # further check that for groups representing list the index of items is either
    # provided via item_index attribute or can be read from name of item
    not_so_homogenous_list = [ 1, 2, 3, [4],5 ,6 ]
    h_dataset,subitems = load_builtins.create_listlike_dataset(not_so_homogenous_list,h5_data,"not_so_homogenous_list",**compression_kwargs)
    assert isinstance(h_dataset,h5.Group)
    item_name = "data{:d}"
    index = -1 
    loaded_list = load_builtins.ListLikeContainer(h_dataset.attrs,b'list',list)
    subitems1,subitems2 = itertools.tee(subitems,2)
    index_from_string = load_builtins.ListLikeContainer(h_dataset.attrs,b'list',list)
    for index,(name,item,attrs,kwargs) in enumerate(iter(subitems1)):
        assert item_name.format(index) == name and item == not_so_homogenous_list[index]
        assert attrs == {"item_index":index} and kwargs == compression_kwargs
        if isinstance(item,list):
            item_dataset,_ = load_builtins.create_listlike_dataset(item,h_dataset,name,**compression_kwargs)
        else:
            item_dataset = h_dataset.create_dataset(name,data = item)
        item_dataset.attrs.update(attrs)
        loaded_list.append(name,item,item_dataset.attrs)
        index_from_string.append(name,item,{})
    assert index + 1 == len(not_so_homogenous_list)
    assert loaded_list.convert() == not_so_homogenous_list
    assert index_from_string.convert() == not_so_homogenous_list

    # check that list groups which do not provide num_items attribute 
    # are automatically expanded to properly cover the highest index encountered
    # for any of the list items.
    no_num_items = {key:value for key,value in h_dataset.attrs.items() if key != "num_items"}
    no_num_items_container = load_builtins.ListLikeContainer(no_num_items,b'list',list)
    for index,(name,item,attrs,kwargs) in enumerate(iter(subitems2)):
        assert item_name.format(index) == name and item == not_so_homogenous_list[index]
        assert attrs == {"item_index":index} and kwargs == compression_kwargs
        item_dataset = h_dataset.get(name,None)
        no_num_items_container.append(name,item,{})
    assert index + 1 == len(not_so_homogenous_list)
    assert no_num_items_container.convert() == not_so_homogenous_list

    # check that list the first of which is not a scalar is properly mapped
    # to a group. Also check that ListLikeContainer.append raises exception
    # in case neither item_index is provided nor an index value can be parsed
    # from the tail of its name. Also check that ListLikeContainer.append
    # raises exception in case value for item_index already has been loaded
    object_list = [ [4, 5 ] ,6, [ 1, 2, 3 ] ]
    h_dataset,subitems = load_builtins.create_listlike_dataset(object_list,h5_data,"object_list",**compression_kwargs)
    assert isinstance(h_dataset,h5.Group)
    item_name = "data{:d}"
    wrong_item_name = item_name + "_ni"
    index = -1
    loaded_list = load_builtins.ListLikeContainer(h_dataset.attrs,b'list',list)
    index_from_string = load_builtins.ListLikeContainer(h_dataset.attrs,b'list',list)
    for index,(name,item,attrs,kwargs) in enumerate(iter(subitems)):
        assert item_name.format(index) == name and item == object_list[index]
        assert attrs == {"item_index":index} and kwargs == compression_kwargs
        if isinstance(item,list):
            item_dataset,_ = load_builtins.create_listlike_dataset(item,h_dataset,name,**compression_kwargs)
        else:
            item_dataset = h_dataset.create_dataset(name,data = item)
        item_dataset.attrs.update(attrs)
        loaded_list.append(name,item,item_dataset.attrs)
        with pytest.raises(KeyError,match = r"List\s+like\s+item name\s+'\w+'\s+not\s+understood"):
            index_from_string.append(wrong_item_name.format(index),item,{})
        # check that previous error is not triggered when
        # legacy 4.0.x loader injects the special value helpers.nobody_is_my_name which
        # is generated by load_nothing function. this is for example used as load method
        # for legacy 4.0.x np.masked.array objects where the mask is injected in parallel
        # in the root group of the corresponding values data set. By silently ignoring
        # this special value returned by load_nothing it can be assured that for example
        # mask datasets of numpy.masked.array objects hickup the loader.
        index_from_string.append(wrong_item_name.format(index),helpers.nobody_is_my_name,{})
        if index < 1:
            continue
        with pytest.raises(IndexError, match = r"Index\s+\d+\s+already\s+set"):
            loaded_list.append(name,item,{"item_index":index-1})
    assert index + 1 == len(object_list)

    # assert that list of strings where first string has length 1 is properly mapped
    # to group
    string_list = test_set = ['I','confess','appriciate','hickle','times']
    h_dataset,subitems = load_builtins.create_listlike_dataset(string_list,h5_data,"string_list",**compression_kwargs)
    assert isinstance(h_dataset,h5.Group)
    item_name = "data{:d}"
    index = -1 
    loaded_list = load_builtins.ListLikeContainer(h_dataset.attrs,b'list',list)
    index_from_string = load_builtins.ListLikeContainer(h_dataset.attrs,b'list',list)
    for index,(name,item,attrs,kwargs) in enumerate(iter(subitems)):
        assert item_name.format(index) == name and item == string_list[index]
        assert attrs == {"item_index":index} and kwargs == compression_kwargs
        item_dataset = h_dataset.create_dataset(name,data = item)
        item_dataset.attrs.update(attrs)
        loaded_list.append(name,item,item_dataset.attrs)
        index_from_string.append(name,item,{})
    assert index + 1 == len(string_list)
    assert loaded_list.convert() == string_list
    assert index_from_string.convert() == string_list

    # assert that list which contains numeric values and strings is properly mapped
    # to group
    mixed_string_list = test_set = [12,2.8,'I','confess','appriciate','hickle','times']
    h_dataset,subitems = load_builtins.create_listlike_dataset(mixed_string_list,h5_data,"mixed_string_list",**compression_kwargs)
    assert isinstance(h_dataset,h5.Group)
    item_name = "data{:d}"
    index = -1 
    loaded_list = load_builtins.ListLikeContainer(h_dataset.attrs,b'list',list)
    index_from_string = load_builtins.ListLikeContainer(h_dataset.attrs,b'list',list)
    for index,(name,item,attrs,kwargs) in enumerate(iter(subitems)):
        assert item_name.format(index) == name and item == mixed_string_list[index]
        assert attrs == {"item_index":index} and kwargs == compression_kwargs
        item_dataset = h_dataset.create_dataset(name,data = item)
        item_dataset.attrs.update(attrs)
        loaded_list.append(name,item,item_dataset.attrs)
        index_from_string.append(name,item,{})
    assert index + 1 == len(mixed_string_list)
    assert loaded_list.convert() == mixed_string_list
    assert index_from_string.convert() == mixed_string_list
    

def test_set_container(h5_data,compression_kwargs):
    """
    tests storing and loading of set
    """

    # check that set of strings is store as group
    test_set = {'I','confess','appriciate','hickle','times'}
    h_setdataset,subitems = load_builtins.create_setlike_dataset(test_set,h5_data,"test_set",**compression_kwargs)
    set_container = load_builtins.SetLikeContainer(h_setdataset.attrs,b'set',set)
    for name,item,attrs,kwargs in subitems:
        set_container.append(name,item,attrs)
    assert set_container.convert() == test_set

    # check that set of single bytes is stored as single dataset
    test_set_2 = set(b"hello world")
    h_setdataset,subitems = load_builtins.create_setlike_dataset(test_set_2,h5_data,"test_set_2",**compression_kwargs)
    assert isinstance(h_setdataset,h5.Dataset) and set(h_setdataset[()]) == test_set_2
    assert not subitems and iter(subitems)
    assert load_builtins.load_list_dataset(h_setdataset,b'set',set) == test_set_2

    # check that set containing byte strings is stored as group
    test_set_3 = set((item.encode("utf8") for item in test_set))
    h_setdataset,subitems = load_builtins.create_setlike_dataset(test_set_3,h5_data,"test_set_3",**compression_kwargs)
    set_container = load_builtins.SetLikeContainer(h_setdataset.attrs,b'set',set)
    for name,item,attrs,kwargs in subitems:
        set_container.append(name,item,attrs)
    assert set_container.convert() == test_set_3

    # check that empty set is represented by empty dataset
    h_setdataset,subitems = load_builtins.create_setlike_dataset(set(),h5_data,"empty_set",**compression_kwargs)
    assert isinstance(h_setdataset,h5.Dataset) and h_setdataset.size == 0
    assert not subitems and iter(subitems)
    assert load_builtins.load_list_dataset(h_setdataset,b'set',set) == set()
    

def test_dictlike_dataset(h5_data,compression_kwargs):
    """
    test storing and loading of dict
    """

    class KeyClass():
        """class used as dict key"""

    allkeys_dict = {
        'string_key':0,
        b'bytes_key':1,
        12:2,
        0.25:3,
        complex(1,2):4,
        None:5,
        (1,2,3):6,
        KeyClass():7,
        KeyClass:8
    }

    # check that dict is stored as group
    # check that string and byte string keys are mapped to dataset or group name 
    # check that scalar dict keys are converted to their string representation
    # check that for all other keys a key value pair is created
    h_datagroup,subitems = load_builtins.create_dictlike_dataset(allkeys_dict,h5_data,"allkeys_dict",**compression_kwargs)
    assert isinstance(h_datagroup,h5.Group)
    invalid_key = b''
    last_entry = -1
    load_dict = load_builtins.DictLikeContainer(h_datagroup.attrs,b'dict',dict)
    ordered_dict = collections.OrderedDict()
    for name,item,attrs,kwargs in subitems:
        value = item
        if attrs["key_base_type"] == b"str":
            key = name[1:-1]
        elif attrs["key_base_type"] == b"bytes":
            key = name[2:-1].encode("utf8")
        elif attrs["key_base_type"] == b'key_value':
            key = item[0]
            value = item[1]
        else: 
            load_key = load_builtins.dict_key_types_dict.get(attrs["key_base_type"],None)
            if load_key is None:
                raise ValueError("key_base_type '{}' invalid".format(attrs["key_base_type"]))
            key = load_key(name)
        assert allkeys_dict.get(key,invalid_key) == value
        load_dict.append(name,item,attrs)
        last_entry = attrs.get("key_idx",None)
        ordered_dict[key] = value
    assert last_entry + 1 == len(allkeys_dict) 
    assert load_dict.convert() == allkeys_dict

    # verify that DictLikeContainer.append raises error in case invalid key_base_type
    # is provided
    with pytest.raises(ValueError, match = r"key\s+type\s+'.+'\s+not\s+understood"):
        load_dict.append("invalid_key_type",12,{"key_idx":9,"key_base_type":b"invalid_type"})
    tuple_key = ('a','b','c')

    # verify that DictLikeContainer.append raises error in case index of key value pair
    # within dict is whether provided by key_index attribute nor can be parsed from 
    # name of corresponding dataset or group
    with pytest.raises(KeyError, match = r"invalid\s+dict\s+item\s+key_index\s+missing"):
        load_dict.append(str(tuple_key),9,{"item_index":9,"key_base_type":b"tuple"})

    # check that helpers.nobody_is_my_name injected for example by load_nothing is silently
    # ignored in case no key could be retrieved from dataset or sub group
    load_dict.append(
        str(tuple_key), helpers.nobody_is_my_name,
        {"item_index":9,"key_base_type":b"tuple"}
    )
    with pytest.raises(KeyError):
        assert load_dict.convert()[tuple_key] is None

    # check that if key_idx attribute is provided key value pair may be added
    load_dict.append(str(tuple_key),9,{"key_idx":9,"key_base_type":b"tuple"})
    assert load_dict.convert()[tuple_key] == 9

    # verify that DictLikeContainer.append raises error in case item index already
    # set
    with pytest.raises(IndexError,match = r"Key\s+index\s+\d+\s+already\s+set"):
        load_dict.append(str(tuple_key),9,{"key_idx":9,"key_base_type":b"tuple"})

    # check that order of OrderedDict dict keys is not altered on loading data from 
    # hickle file
    h_datagroup,subitems = load_builtins.create_dictlike_dataset(ordered_dict,h5_data,"ordered_dict",**compression_kwargs)
    assert isinstance(h_datagroup,h5.Group)
    last_entry = -1
    load_ordered_dict = load_builtins.DictLikeContainer(h_datagroup.attrs,b'dict',collections.OrderedDict)
    for name,item,attrs,kwargs in subitems:
        value = item
        if attrs["key_base_type"] == b"str":
            key = name[1:-1]
        elif attrs["key_base_type"] == b"bytes":
            key = name[2:-1].encode("utf8")
        elif attrs["key_base_type"] == b'key_value':
            key = item[0]
            value = item[1]
        else: 
            load_key = load_builtins.dict_key_types_dict.get(attrs["key_base_type"],None)
            if load_key is None:
                raise ValueError("key_base_type '{}' invalid".format(attrs["key_base_type"]))
            key = load_key(name)
        assert ordered_dict.get(key,invalid_key) == value
        load_ordered_dict.append(name,item,attrs)
        last_entry = attrs.get("key_idx",None)
    assert last_entry + 1 == len(allkeys_dict) 
    assert load_ordered_dict.convert() == ordered_dict


# %% MAIN SCRIPT
if __name__ == "__main__":
    from _pytest.fixtures import FixtureRequest
    from hickle.tests.conftest import compression_kwargs
    for h5_root,keywords in (
        ( h5_data(request),compression_kwargs(request) )
        for request in (FixtureRequest(test_scalar_dataset),)
    ):
        test_scalar_dataset(h5_root,keywords)
    for h5_root,keywords in (
        ( h5_data(request),compression_kwargs(request) )
        for request in (FixtureRequest(test_non_dataset),)
    ):
        test_non_dataset(h5_root,keywords)
    for h5_root,keywords in (
        ( h5_data(request),compression_kwargs(request) )
        for request in (FixtureRequest(test_listlike_dataset),)
    ):
        test_listlike_dataset(h5_root,keywords)
    for h5_root,keywords in (
        ( h5_data(request),compression_kwargs(request) )
        for request in (FixtureRequest(test_set_container),)
    ):
        test_set_container(h5_root,keywords)
    for h5_root,keywords in (
        ( h5_data(request),compression_kwargs(request) )
        for request in (FixtureRequest(test_dictlike_dataset),)
    ):
        test_dictlike_dataset(h5_root,keywords)

    
    

