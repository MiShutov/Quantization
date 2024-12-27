import os
from torch.utils.tensorboard import SummaryWriter

class LoggerPTQ:
    def __init__(self, logdir):
        self.logdir = logdir
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        self.set_version_dir()
        self.writer = SummaryWriter(log_dir=self.version_dir)
        self.block_label = None
        
       
    def set_version_dir(self):
        logdir_files = os.listdir(self.logdir)
        last_v = -1
        for file in logdir_files:
            if 'verison_' in file:
                v = int(file.split('verison_')[-1])
                if v > last_v:
                    last_v = v
        new_version = last_v+1
        self.version_dir = f"{self.logdir}/verison_{new_version}"
        os.makedirs(self.version_dir, exist_ok=False)


    def log_scalar(self, mode, scalar_name, scalar, step):
        assert self.block_label
        self.writer.add_scalar(
            f"{self.block_label}/{mode}/{scalar_name}", scalar, step
        )
        self.writer.flush()
