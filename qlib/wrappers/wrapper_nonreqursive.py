from nip import nip
import torch
import torch.nn as nn


class Wrapper_:
    def __init__(self, wrap_rule):
        self.wrap_rule = wrap_rule


    def check_exception(self, module_full_name):
        for excepion in self.wrap_rule['exceptions']:
            if excepion in module_full_name:
                return excepion
        return False


    def wrap_model(self, current_module, prefix=''):
        exceptions = self.wrap_rule['exceptions']
        for module_name, module in current_module.named_children():
            full_name = f"{prefix}.{module_name}" if prefix else module_name
            module_class_name = str(module.__class__.__name__)
            excepion = self.check_exception(full_name)
            if (module_class_name in self.wrap_rule) and not excepion:
                new_module = self.wrap_rule[module_class_name].wrap_module(module)
                setattr(current_module, module_name, new_module)
                print(full_name, 'wrapped!')
            elif excepion:
                new_module = exceptions[excepion].wrap_module(module)
                setattr(current_module, module_name, new_module)
                print(full_name, 'wrapped! (exception)')
            else:
                self.wrap_model(module, full_name)


