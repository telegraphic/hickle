def create_package_test(myclass_type,h_group,name,**kwargs):
    return h_group,()

def load_package_test(h_node,base_type,py_obj_type):
    return {12:12}


class_register = [
    ( dict,b'dict',create_package_test,load_package_test )
]

exclude_register = [b'please_kindly_ignore_me']
