# names all optional loaders defined by any load_*.py file
# will be extended by any optional loader managed by hickle
# core engine. Names of optional_loaders must be all lower case.
# Corresponding option attributes in hickle file will be all
# upper case.
optional_loaders = {
    # option loader for defining custom loader methods and 
    # PyContainer classes. By marking them as custom option
    # they are only activate if specified by a call to 
    # hickle.dump. If not specified than custom objects and
    # classes will simply be stored as pickle string.
    # The data may in this case not be recoverable if 
    # underlying classes are not available or not compatible
    # any more due to disruptive changes. When dumped using
    # custom loader hickle at least can try to restore data
    # as numpy.array or python dict like structure with metadata
    # attached as is for further inspection.
    'custom',
}

# prefix for optional_loaders attribute names which are all
# uppercase
attribute_prefix = "OPTION_"
