import torch
from qlib.quant_layers.qlinear import QLinear
from qlib.processing.tensor2d_group_scaling import Tensor2dGroupScaling


class WeightScalingQLinearForDevice(QLinear):
    '''
    QLinear - is a main class ...
    '''
    def __init__(
            self,
            weight_shape,
            bias,
            weight_scaling_params,
            weight_quantizer_params,
            input_quantizer_params=None,
            output_quantizer_params=None,
            matmul_out_scale=None,
        ):
        super().__init__(
            weight_shape=weight_shape,
            bias=bias,
            input_quantizer_params=input_quantizer_params,
            output_quantizer_params=output_quantizer_params,
            weight_quantizer_params=weight_quantizer_params,
        )
        if matmul_out_scale is not None:
            self.register_buffer(
                "matmul_out_scale",
                torch.tensor(1 / matmul_out_scale, dtype=torch.float32, requires_grad=False)
            )

        weight_scaling_params["tensor_shape"] = weight_shape
        self.weight_scaling = Tensor2dGroupScaling(**weight_scaling_params)
        
        compressed_weight_shape, compressed_weight_dtype = self.weight_quantizer.get_compressed_weight_params(self.weight_shape)
        self.register_buffer(
            "compressed_weight",
            torch.empty(
                compressed_weight_shape,
                dtype=compressed_weight_dtype,
                requires_grad=False
            )
        )


    @torch.no_grad()
    def configure(self, fp_module, device, verbose=False, *args, **kwargs):
        w_fp = fp_module.weight.data.to(device)
        self.to(device)

        if hasattr(self, "matmul_out_scale"):
            self.weight_scaling.configure(w_fp / self.matmul_out_scale)        
            w_fp_scaled = self.weight_scaling.forward(w_fp / self.matmul_out_scale)
        else:
            self.weight_scaling.configure(w_fp)
            w_fp_scaled = self.weight_scaling.forward(w_fp)

        compressed_weight = self.weight_quantizer.quantize(w_fp_scaled)
        self.compressed_weight.data.copy_(compressed_weight)
        
        if verbose:
            self.check_mse(w_fp, kwargs.get('module_name', None))


    def weight_transform(self, w):
        return self.weight_scaling.forward(w)


    @property
    def weight(self):
        """
        Attention! Use this method only for debugging / getting pseudoquant model without activation quantization.
        Do not this method during your inference / training pipeline.
        """
        decompressed_weight = self.weight_quantizer.dequantize(self.compressed_weight)
        w = self.weight_scaling.inverse(decompressed_weight)
        if hasattr(self, "matmul_out_scale"):
            w = w * self.matmul_out_scale
        return w


    def forward(self, x):
        # x_dtype = x.dtype
        # decompressed_weight = self.weight_quantizer.dequantize(self.compressed_weight)
        # w = self.weight_scaling.inverse(decompressed_weight)
        # x = torch.nn.functional.linear(x, w)
        # if hasattr(self, "matmul_out_scale"):
        #     x = x * self.matmul_out_scale
        # return x.to(x_dtype)

        x_dtype = x.dtype

        x_int8, x_scales = self.input_quantizer(x, simulation=False)
        x_scales = x_scales[0, :, 0]
        w_int8 = self.compressed_weight
        w_scales = (self.weight_scaling.scale.data[:, 0]).to(torch.float16)

        x = torch.nn.functional.linear(x_int8.float(), w_int8.float())
        if hasattr(self, "matmul_out_scale"):
            x = x * self.matmul_out_scale
        
        x = x.to(torch.float16)
        assert torch.isfinite(x).all()

        x = x * w_scales * x_scales[:, None]

        x = self.output_quantizer(x)
        return x.to(x_dtype)