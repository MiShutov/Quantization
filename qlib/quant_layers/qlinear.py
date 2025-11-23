from abc import ABC, abstractmethod
import torch

from qlib.utils.imports import setup_module
from qlib.quantizers.fake_activation_quantizer import FakeActivationQuantizer


class QLinear(torch.nn.Module, ABC):
    '''
    QLinear - is a main class ...
    '''
    def __init__(
            self,
            weight_shape,
            bias,
            weight_quantizer_params,
            input_quantizer_params=None,
            output_quantizer_params=None,
        ):
        super().__init__()
        self.weight_shape = weight_shape
        self.bias = bias
        assert self.bias == False

        # Input Quantizer
        if input_quantizer_params is None:
            self.input_quantizer = FakeActivationQuantizer()
        else:
            self.input_quantizer = setup_module(**input_quantizer_params)
            
        # Output Quantizer
        if output_quantizer_params is None:
            self.output_quantizer = FakeActivationQuantizer()
        else: 
            self.output_quantizer = setup_module(**output_quantizer_params)
        
        # Weight Quantizer
        self.weight_quantizer = setup_module(**weight_quantizer_params)


    @abstractmethod
    def configure(self):
        pass


    @property
    @abstractmethod
    def weight(self):
        pass


    @abstractmethod
    def forward(self, x):
        pass


    @torch.no_grad()
    def check_mse(self, w_fp, module_name=None):
        w_q = self.weight
        err = torch.mean(((w_fp.cpu() - w_q.cpu()) / (w_fp.cpu().std() + 1e-8))**2)
        if module_name is not None:
            print(module_name)
        print(f"error (w-wq)^2: {err.mean():.3f}")