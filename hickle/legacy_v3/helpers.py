import re
import six

def get_type_and_data(h_node):
    """ Helper function to return the py_type and data block for a HDF node """
    py_type = h_node.attrs["type"][0]
    data = h_node[()]
#    if h_node.shape == ():
#        data = h_node.value
#    else:
#        data  = h_node[:]
    return py_type, data

def get_type(h_node):
    """ Helper function to return the py_type for a HDF node """
    py_type = h_node.attrs["type"][0]
    return py_type

def sort_keys(key_list):
    """ Take a list of strings and sort it by integer value within string

    Args:
        key_list (list): List of keys

    Returns:
        key_list_sorted (list): List of keys, sorted by integer
    """

    # Py3 h5py returns an irritating KeysView object
    # Py3 also complains about bytes and strings, convert all keys to bytes
    if six.PY3:
        key_list2 = []
        for key in key_list:
            if isinstance(key, str):
                key = bytes(key, 'ascii')
            key_list2.append(key)
        key_list = key_list2

    # Check which keys contain a number
    numbered_keys = [re.search(br'\d+', key) for key in key_list]

    # Sort the keys on number if they have it, or normally if not
    if(len(key_list) and not numbered_keys.count(None)):
        to_int = lambda x: int(re.search(br'\d+', x).group(0))
        return(sorted(key_list, key=to_int))
    else:
        return(sorted(key_list))


def check_is_iterable(py_obj):
    """ Check whether a python object is iterable.

    Note: this treats unicode and string as NON ITERABLE

    Args:
        py_obj: python object to test

    Returns:
        iter_ok (bool): True if item is iterable, False is item is not
    """
    if six.PY2:
        string_types = (str, unicode)
    else:
        string_types = (str, bytes, bytearray)
    if isinstance(py_obj, string_types):
        return False
    try:
        iter(py_obj)
        return True
    except TypeError:
        return False


def check_is_hashable(py_obj):
    """ Check if a python object is hashable

    Note: this function is currently not used, but is useful for future
          development.

    Args:
        py_obj: python object to test
    """

    try:
        py_obj.__hash__()
        return True
    except TypeError:
        return False


def check_iterable_item_type(iter_obj):
    """ Check if all items within an iterable are the same type.

    Args:
        iter_obj: iterable object

    Returns:
        iter_type: type of item contained within the iterable. If
                   the iterable has many types, a boolean False is returned instead.

    References:
    http://stackoverflow.com/questions/13252333/python-check-if-all-elements-of-a-list-are-the-same-type
    """
    iseq = iter(iter_obj)

    try:
        first_type = type(next(iseq))
    except StopIteration:
        return False
    except Exception as ex:
        return False
    else:
        return first_type if all((type(x) is first_type) for x in iseq) else False
