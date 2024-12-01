from nip import nip
import torch
import torch.nn as nn


class Wrapper:
    def __init__(self, wrap_rule, exceptions={}):
        self.wrap_rule = wrap_rule
        self.exceptions = exceptions


    def check_exception(self, module_class_name, module_full_name):
        class_exceptions = self.exceptions.get(module_class_name, False)
        if class_exceptions:
            for excepion in class_exceptions:
                if excepion in module_full_name:
                    return class_exceptions[excepion]
        return False


    def wrap_model(self, current_module, prefix=''):
        for module_name, module in current_module.named_children():
            full_name = f"{prefix}.{module_name}" if prefix else module_name
            module_class_name = str(module.__class__.__name__)
            
            if module_class_name in self.wrap_rule:
                exception = self.check_exception(module_class_name, full_name)
                if not exception:
                    new_module = self.wrap_rule[module_class_name].wrap_module(module)
                    setattr(current_module, module_name, new_module)
                    #print(full_name, 'wrapped!')
                else:
                    new_module = exception.wrap_module(module)
                    setattr(current_module, module_name, new_module)
                    #print(full_name, 'wrapped! (exception)')
            else:
                self.wrap_model(module, full_name)


