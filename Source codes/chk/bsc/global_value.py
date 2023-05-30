import copy
from collections import OrderedDict

def init():
    global _global_dict
    _global_dict = OrderedDict()
    
def set_value(key, value):
    _global_dict[key] = value

def get_value(key):
    if key in _global_dict:
        return _global_dict[key]
    else:
        return None

def get_global_value():
    return copy.deepcopy(_global_dict)
    