import collections

def map_nested_dicts(ob, func):
    if isinstance(ob, collections.Mapping):
        return {k: map_nested_dicts(v, func) for k, v in ob.items()}
    else:
        return func(ob)

def map_nested_dicts_modify(ob, func):
    for k, v in ob.items():
        if isinstance(v, collections.Mapping):
            map_nested_dicts_modify(v, func)
        else:
            ob[k] = func(v)

def map_nested_dicts_modify_with_other(ob, ob_other, func):
    for k, v in ob.items():
        if isinstance(v, collections.Mapping):
            map_nested_dicts_modify(v, ob_other[k], func)
        else:
            ob[k] = func(v, ob_other[k])