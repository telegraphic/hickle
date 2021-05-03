import pytest
import sys
import types
import functools as ft
import threading
import os
import os.path
import importlib
import collections
import ctypes
import re

# list of function names which shall not be
# traced when compression keyword hardening
# test run is executed
non_core_loader_functions = {
    'type_legacy_mro',
    'load_pickled_data',
    'recover_custom_dataset',
    #'register_compact_expand',
    '_moc_numpy_array_object_lambda',
    'fix_lambda_obj_type',
    'LoaderManager.load_loader',
    'RecoverGroupContainer.convert',
    'NoContainer.convert',
    '_DictItemContainer.convert',
    'ExpandReferenceContainer.convert',
    'RecoverGroupContainer.filter',
    'ExpandReferenceContainer.filter',
    'ReferenceManager.resolve_type',
    'RecoverGroupContainer._append'
}

def pytest_addoption(parser):
    """
    adds enable_compression keywort to pytest commandline options
    for enabling h5py compression keyword hardening testing of
    dump functions of hikcle.loaders  and hickle core loaders
    """
    parser.addoption(
        "--enable-compression",
        action='store',
        nargs='?',
        const=6,
        type=int,
        choices=range(0,10),
        help="run all tests with bzip compression enabled. Optionally specify compression level 0-9 (default 6)",
        dest="enable_compression"
    )

def _get_trace_function(trace_function):
    """
    try to get hold of FunctionType object of passed in Method, Function or callable
    """
    while not isinstance(trace_function,(types.FunctionType,types.LambdaType,types.BuiltinFunctionType)):
        if isinstance(trace_function,(types.MethodType,types.BuiltinMethodType)):
            trace_function = getattr(trace_function,'__func__')
            continue
        if isinstance(trace_function,ft.partial):
            trace_function = trace_function.func
            continue
        return (
            getattr(trace_function,'__call__',trace_function) 
            if callable(trace_function) and not isinstance(trace_function,type) else
            None
        )
    return trace_function

# keyword arguments to yield from compression_kwargs fixture below
# may in future become a list of dictionaries to be yieled for
# running same test with different sets of compression keywors
# (implizit parametrization of tests)
_compression_args =  dict(
    compression='gzip',
    compression_opts=6
)
_test_compression = None

def pytest_configure(config):
    """
    make no_compression mark available from pytest.mark.
    if not yet activated enable profiling of dump methods and functions
    and set compression_level selected on commandline if explicitly
    specified.
    """
    global _test_compression

    config.addinivalue_line(
        "markers","no_compression: do not enforce h5py comression hardening testing"
    )
    if _test_compression is not None:
        return
    compression_level = config.getoption("enable_compression",default=-1)
    if compression_level is None or compression_level < 0:
        return
    _compression_args['compression_opts'] = compression_level
    _test_compression = True

# local handle of no_compression mark
no_compression = pytest.mark.no_compression

@pytest.fixture#(scope='session')
def compression_kwargs(request):
    """
    fixture providing the compressoin related keyword arguments
    to be passed to any test not marked with no_compression mark
    and expecting compression_kwargs as one of its parameters 
    """
    global _test_compression
    yield ( _compression_args if _test_compression else {} )

# list of distinct copyies of LoaderManager.register_class function
# keys are either "<filename>::LoaderManager.register_class" or
# copy of code object executed when LoaderManager.register_class method
# is called
_trace_register_class = {}

# list of dump_functions to be traced with respect to beeing
# passed the compression related keywords provided throug compression_kwargs
# fixture above. In case a call to any of theses does not include at least these
# keywords an AssertionError Exception is raised.
_trace_functions = collections.OrderedDict()

# profiling function to be called after execution of _trace_loader_funcs
# below
_trace_profile_call = None

# index of dump_function argument in argument list of LoaderManager.register_class
# method. 
_trace_function_argument_default = -1
def _chain_profile_call(frame,event,arg):
    global _trace_profile_call
    if _trace_profile_call:
        next_call = _trace_profile_call(frame,event,arg)
        if next_call:
            _trace_profile_call = next_call

# argument names which correspond to argument beeing passed dump_function
# object
_trace_function_arg_names = {'dump_function'}


# the pytest session tracing of proper handling of compression related
# keywords is activated for
_traced_session = None

_loader_file_pattern = re.compile(r'^load_\w+\.py$')

def pytest_sessionstart(session):
    """
    pytest hook called at start of session.
    - collects all functions exported by hickle.lookup module (for now) and
      records inserts "<filename>::<function.__qualname__>" strings into
      _trace_functions list for any not listed in above non_core_loader_functions
    - collects all dump_functions listed in class_register tables of all
      hickle.loaders.load_*.py modules.
    """
    global _test_compression,_traced_session,_trace_register_class,_trace_functions,_trace_profile_call
    if _test_compression is None:
        pytest_configure(session.config)
    if not _test_compression:
        return None
    # extract all loader function from hickle.lookup
    lookup_module = sys.modules.get('hickle.lookup',None)
    if not isinstance(lookup_module,types.ModuleType):
        lookup_module_spec = importlib.util.find_spec("hickle.lookup")
        
        lookup_module = importlib.util.module_from_spec(lookup_module_spec)
        lookup_module_spec.loader.exec_module(lookup_module)
    register_class = lookup_module.LoaderManager.register_class
    register_class_code = register_class.__func__.__code__
    trace_function_argument = register_class_code.co_argcount + register_class_code.co_kwonlyargcount
    for argid,trace_function in ( (count,varname) for count,varname in enumerate(register_class_code.co_varnames[:(register_class_code.co_argcount + register_class_code.co_kwonlyargcount)]) if varname in _trace_function_arg_names ):
        trace_function_argument = argid
        break
    if trace_function_argument < 0:
        return None
    _trace_function_argument_default = trace_function_argument
    qualname = getattr(register_class,'__qualname__',register_class.__name__)
    code_name = qualname if qualname.rsplit('.',1) == register_class_code.co_name else register_class_code.co_name
    _trace_register_class.update({"{}::{}".format(register_class_code.co_filename,code_name):trace_function_argument})
    for loader_func_name,loader_func in ( 
        (func_name,func) 
        for name, item in lookup_module.__dict__.items() 
        if isinstance(item,(types.FunctionType,type))
        for func_name,func in ( 
            ((name,item),)
            if isinstance(item,types.FunctionType) else 
            ( 
                ( meth_name,meth) 
                for meth_name,meth in item.__dict__.items() 
                if isinstance(meth,types.FunctionType) 
            ) 
        )
        if func_name[:2] != '__' and func_name[-2:] != '__' 
    ):
        loader_func = _get_trace_function(loader_func)
        if loader_func is not None and loader_func.__module__ == lookup_module.__name__:
            code = loader_func.__code__
            qualname = getattr(loader_func,'__qualname__',loader_func.__name__) 
            if qualname not in non_core_loader_functions:
                code_name = qualname if qualname.rsplit('.',1) == code.co_name else code.co_name
                _trace_functions["{}::{}".format(code.co_filename,code_name)] = (loader_func.__module__,qualname)
    # extract all dump functions from any known loader module
    hickle_loaders_path = os.path.join(os.path.dirname(lookup_module.__file__),'loaders')
    for loader in os.scandir(hickle_loaders_path):
        if not loader.is_file() or _loader_file_pattern.match(loader.name) is None:
            continue
        loader_module_name = "hickle.loaders.{}".format(loader.name.rsplit('.',1)[0])
        loader_module = sys.modules.get(loader_module_name,None)
        if loader_module is None:
            
            loader_module_spec = importlib.util.find_spec("hickle.loaders.{}".format(loader.name.rsplit('.',1)[0]))
            if loader_module_spec is None:
                continue
            loader_module = importlib.util.module_from_spec(loader_module_spec)
            try:
                loader_module_spec.loader.exec_module(loader_module)
            except ModuleNotFoundError:
                continue
            except ImportError:
                if sys.version_info[0] > 3 or sys.version_info[1] > 5:
                    raise
                continue
        class_register_table = getattr(loader_module,'class_register',())
        # trace function has cls/self
        for dump_function in ( entry[trace_function_argument-1] for entry in class_register_table ):
            dump_function = _get_trace_function(dump_function)
            if dump_function is not None:
                code = dump_function.__code__
                qualname = getattr(dump_function,'__qualname__',dump_function.__name__)
                code_name = qualname if qualname.rsplit('.',1) == code.co_name else code.co_name
                _trace_functions["{}::{}".format(code.co_filename,code_name)] = (dump_function.__module__,qualname)
    # activate compression related profiling
    _trace_profile_call = sys.getprofile()
    _traced_session = session
    sys.setprofile(_trace_loader_funcs)
    return None

# List of test functions which are marked by no_compression mark
_never_trace_compression = set()

def traceback_from_frame(frame,stopafter):
    """
    helper function used in Python >= 3.7 to beautify traceback
    of AssertionError exceptoin thrown by _trace_loader_funcs
    """
    tb = types.TracebackType(None,frame,frame.f_lasti,frame.f_lineno)
    while frame.f_back is not stopafter.f_back:
        frame = frame.f_back
        tb = types.TracebackType(tb,frame,frame.f_lasti,frame.f_lineno)
    return tb

    
def pytest_collection_finish(session):
    """
    collect all test functions for which comression related keyword monitoring
    shall be disabled.
    """
    if not sys.getprofile() == _trace_loader_funcs:
        return 

    listed = set()
    listemodules = set()
    for item in session.items:
        func = item.getparent(pytest.Function)
        if func not in listed:
            listed.add(func)
            for marker in func.iter_markers(no_compression.name):
                never_trace_code = func.function.__code__
                qualname = getattr(func.function,'__qualname__',func.function.__name__)
                code_name = qualname if qualname.rsplit('.',1) == never_trace_code.co_name else never_trace_code.co_name
                _never_trace_compression.add("{}::{}".format(never_trace_code.co_filename,code_name))
                break
        

def _trace_loader_funcs(frame,event,arg,nochain=False):
    """
    does the actuatual profiling with respect to proper passing compression keywords
    to dump_functions
    """
    global _chain_profile_call, _trace_functions,_never_trace_compression,_trace_register_class,_trace_function_argument_default
    try:
        if event not in {'call','c_call'}:
            return _trace_loader_funcs
        # check if LoaderManager.register_class has been called
        # if get position of dump_function argument and extract
        # code object for dump_function to be registered if not None
        code_block = frame.f_code
        trace_function_argument = _trace_register_class.get(code_block,None)
        if trace_function_argument is not None:
            trace_function = frame.f_locals.get(code_block.co_varnames[trace_function_argument],None)
            load_function = frame.f_locals.get(code_block.co_varnames[trace_function_argument+1],None)
            if load_function is not None:
                load_function = _get_trace_function(load_function)
                _trace_functions.pop("{}::{}".format(load_function.__code__.co_filename,load_function.__code__.co_name),None)
            if trace_function is None:
                return _trace_loader_funcs
            trace_function = _get_trace_function(trace_function)
            if trace_function is None:
                return _trace_loader_funcs
            trace_function_code = getattr(trace_function,'__code__',None)
            if trace_function_code is not None:
                # store code object corresponding to dump_function in _trace_functions list
                # if not yet present there. 
                qualname = getattr(trace_function,'__qualname__',trace_function.__name__)
                code_name = qualname if qualname.rsplit('.',1) == trace_function_code.co_name else trace_function_code.co_name
                trace_function_code_name = "{}::{}".format(trace_function_code.co_filename,code_name)
                if (
                    trace_function_code_name not in _trace_register_class and
                    ( 
                        trace_function_code_name not in _trace_functions or
                        trace_function_code not in _trace_functions
                    )
                ):
                    trace_function_spec = (trace_function.__module__,qualname)
                    _trace_functions[trace_function_code] = trace_function_spec
                    _trace_functions[trace_function_code_name] = trace_function_spec
            return _trace_loader_funcs
        # estimate qualname from local variable stored in frame.f_local corresponding
        # to frame.f_code.co_varnames[0] if any.
        object_self_name = frame.f_code.co_varnames[:1]
        if object_self_name:
            self = frame.f_locals.get(object_self_name[0],None)
            module = getattr(self,'__module__','')
            if isinstance(module,str) and module.split('.',1)[0] == 'hickle' and isinstance(getattr(self,'__name__',None),str):
                method = getattr(self,frame.f_code.co_name,None)
                if method is not None and getattr(method,'__code__',None) == frame.f_code:
                    code_name = "{}::{}.{}".format(
                        frame.f_code.co_filename,
                        getattr(self,'__qualname__',self.__name__),
                        frame.f_code.co_name
                    )
                else:
                    code_name = "{}::{}".format(frame.f_code.co_filename,frame.f_code.co_name)
            else:
                code_name = "{}::{}".format(frame.f_code.co_filename,frame.f_code.co_name)
        else:
            code_name = "{}::{}".format(frame.f_code.co_filename,frame.f_code.co_name)
        # check if frame could encode a clall to a new incarnation of LoaderManager.register_class
        # method. Add its code object to the list of known incarnations and rerun above code
        if code_block.co_name == 'register_class':
            trace_function_argument = _trace_register_class.get(code_name,None)
            if trace_function_argument is not None:
                _trace_register_class[code_block] = trace_function_argument
                return _trace_loader_funcs(frame,event,arg,True)
            if (
                code_block.co_filename.rsplit('/',2) == ['hickle','lookup.py'] and 
                code_block.co_varnames > trace_function_argument and 
                code_block.co_varnames[_trace_function_argument_default] in _trace_function_arg_names
            ):
                _trace_register_class[code_name] = _trace_function_argument_default
                _trace_register_class[code_block] = _trace_function_argument_default
                return _trace_loader_funcs(frame,event,arg,True)

        # frame encodes a call to any other function or method. 
        # If the function or method is listed in _trace_functions list check
        # if it received the appropriate set of compresson related keywords
        function_object_spec = _trace_functions.get(frame.f_code,None)
        if function_object_spec is None:
            function_object_spec = _trace_functions.get(code_name,None)
            if function_object_spec is None:
                return _trace_loader_funcs
            _trace_functions[frame.f_code] = function_object_spec
        baseargs = ( 
            (arg,frame.f_locals[arg]) 
            for arg in frame.f_code.co_varnames[:(frame.f_code.co_argcount + frame.f_code.co_kwonlyargcount)]
        )
        kwargs = frame.f_locals.get('kwargs',None)
        if kwargs is not None:
            fullargs = ( (name,arg) for arglist in (kwargs.items(),baseargs) for name,arg in arglist )
        else:
            fullargs = baseargs
        seen_compression_args = set()
        for arg,value in fullargs:
            if arg in seen_compression_args:
                continue
            if _compression_args.get(arg,None) is not None:
                seen_compression_args.add(arg)
                if len(seen_compression_args) == len(_compression_args):
                    return _trace_loader_funcs
        # keywords not passed or filtered prematurely.
        # walk the stack until reaching executed test function.
        # if test function is not marked with no_compression raise
        # AssertionError stating that dump_function did not
        # receive expected compression keywords defined above
        # For Python <= 3.6 collect all functions called between current
        # frame and frame of executed test function. For Python > 3.6 use
        # above traceback_from_frame function to build traceack showing appropriate
        # callstack and context excluding this function to ensure AssertionError
        # exception appears thrown on behlaf of function triggering call encoded by
        # passed frame
        function_object_spec = _trace_functions[frame.f_code]
        if _traced_session is not None:
            test_list = { 
                "{}::{}".format(
                    item.function.__code__.co_filename,
                    getattr(item.function,'__qualname__',
                    item.function.__name__)
                ):item 
                for item in _traced_session.items
            }
            collect_call_tree = []
            next_frame = frame
            while next_frame is not None:
                object_self_name = frame.f_code.co_varnames[:1]
                if object_self_name:
                    self = frame.f_locals.get(object_self_name[0])
                    module = getattr(self,'__module__','')
                    if (
                        isinstance(module,str) and 
                        module.split('.',1)[0] == 'hickle' and 
                        isinstance(getattr(self,'__name__',None),str)
                    ):
                        method = getattr(self,frame.f_code.co_name,None)
                        if method is not None and getattr(method,'__code__',None) == frame.f_code:
                            frame_name = "{}::{}".format(
                                next_frame.f_code.co_filename,
                                getattr(method,'__qualname__',method.__name__)
                            )
                        else:
                            frame_name = "{}::{}".format(next_frame.f_code.co_filename,next_frame.f_code.co_name)
                    else:
                        frame_name = "{}::{}".format(next_frame.f_code.co_filename,next_frame.f_code.co_name)
                else:
                    frame_name = "{}::{}".format(next_frame.f_code.co_filename,next_frame.f_code.co_name)
                if frame_name in _never_trace_compression:
                    return _trace_loader_funcs
                in_test = test_list.get(frame_name,None)
                collect_call_tree.append((next_frame.f_code.co_filename,frame_name,next_frame.f_lineno))
                if in_test is not None:
                    try:
                        tb = traceback_from_frame(frame,next_frame)
                    except TypeError:
                        pass
                    else:
                        raise AssertionError(
                            "'{}': compression_kwargs lost in call".format("::".join(function_object_spec))
                        ).with_traceback(tb)
                    raise AssertionError(
                        "'{}': compression_kwargs lost in call:\n\t{}\n".format(
                            "::".join(function_object_spec),
                            "\n\t".join("{}::{} ({})".format(*call) for call in collect_call_tree[:0:-1])
                        )
                    )
                next_frame = next_frame.f_back
    except AssertionError as ae:
        # check that first entry in traceback does not refer to this function
        if ae.__traceback__.tb_frame.f_code == _trace_loader_funcs.__code__:
            ae.__traceback__ = ae.__traceback__.tb_next
        raise
    #except Exception as e:
    #    import traceback;traceback.print_exc()
    #    import pdb;pdb.set_trace()
    finally:
        if not nochain:
            _chain_profile_call(frame,event,arg)

def pytest_sessionfinish(session):
    sys.setprofile(_trace_profile_call)
