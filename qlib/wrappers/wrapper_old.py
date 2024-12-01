from nip import nip
import torch
import torch.nn as nn


class Wrapper_old:
	def __init__(self, wrap_rule):
		self.wrap_rule = wrap_rule

	def wrap_model(self, model):
		exceptions = self.wrap_rule['exceptions']
		for module_name, module in model.named_children():
			module_class_name = module.__class__.__name__
			for wrap_name in self.wrap_rule:
				if (module_class_name==wrap_name) and (module_name not in exceptions):
					new_module = self.wrap_rule[wrap_name].wrap_module(module)
					setattr(model, module_name, new_module)
					break
				elif module_name in exceptions:
					new_module = self.wrap_rule['exceptions'][module_name].wrap_module(module)
					setattr(model, module_name, new_module)
					break
			else:
				# Recursive process all modules
				self.wrap_model(module)

