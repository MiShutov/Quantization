class HomeQuantWrapper:
    def __init__(self, wrap_rule, exceptions={}):
        self.wrap_rule = wrap_rule
        self.exceptions = exceptions

    def check_exception(self, module_full_name):
        for exception_name in self.exceptions:
            if exception_name in module_full_name:
                return exception_name
        return False

    def wrap_model(self, current_module, prefix=""):
        for module_name, module in current_module.named_children():
            full_name = f"{prefix}.{module_name}" if prefix else module_name
            if module.__class__ in self.wrap_rule:
                if self.check_exception(full_name):
                    if self.exceptions[self.check_exception(full_name)] is not None:
                        new_module = self.exceptions[self.check_exception(full_name)].wrap_module(module, full_name)
                        setattr(current_module, module_name, new_module)
                else:
                    if self.wrap_rule[module.__class__] is not None:
                        new_module = self.wrap_rule[module.__class__].wrap_module(module, full_name)
                        setattr(current_module, module_name, new_module)
            else:
                self.wrap_model(module, full_name)
