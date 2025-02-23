import torch

class ActivationStorage:
	def __init__(
			self,
			train_fp = None,
			train_q = None,
			val_fp = None,
			val_q = None
			):
		if (train_fp is not None) and (train_fp is not None):
			assert len(train_fp) == len(train_q)
		
		if (val_fp is not None) and (val_q is not None):
			assert len(val_fp) == len(val_q)

		self.train_fp = train_fp
		self.train_q = train_q
		self.val_fp = val_fp
		self.val_q = val_q
		
		self.n_train_batches = 0
		self.train_seqlen = None
		if self.train_fp is not None:
			self.n_train_batches = len(self.train_fp)
			self.train_seqlen = self.train_fp[0].shape[0] if self.n_train_batches>0 else None

		self.n_val_batches = 0
		self.val_seqlen = None
		if self.val_fp is not None:
			self.n_val_batches = len(self.val_fp)
			self.val_seqlen = self.val_fp[0].shape[0] if self.n_val_batches>0 else None
		

class HomequantActivationStorage:
	def __init__(
			self,
			mem_block_size = 8 * 2**30, # 8 Gb
			disk_path: str = None
		):
		self.mem_block_size = mem_block_size
		self.disk_path = disk_path


		