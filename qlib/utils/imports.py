import qlib


def setup_module(class_name, params={}):
    module_class = getattr(qlib, class_name)
    return module_class(**params)
