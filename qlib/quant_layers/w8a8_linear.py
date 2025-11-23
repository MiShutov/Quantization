import torch
from qlib.quantizers.dynamic_activation_quantizer import DynamicActivationQuantizer
from qlib.kernels.kernel_int8_int8_matmul import triton_matmul_int8_int8

class MinMaxInitializer:
    def __init__(self):
        pass

    @torch.no_grad()
    def __call__(self, x_grouped, negative_clip, positive_clip, use_offset=False, offset_mode=None):
        x_min = x_grouped.min(axis=-1)[0].unsqueeze(-1).float()
        x_max = x_grouped.max(axis=-1)[0].unsqueeze(-1).float()

        if use_offset == False:
            scale = torch.where(torch.abs(x_min) > torch.abs(x_max), x_min / negative_clip, x_max / positive_clip)
            scale = torch.abs(scale)
            scale = scale.reshape(x_grouped.shape[0], x_grouped.shape[1])
            return scale, None
        else:
            if offset_mode == "float":
                offset = (x_max * negative_clip - x_min * positive_clip) / (positive_clip - negative_clip)
                scale = (x_max + offset) / positive_clip
                scale = torch.abs(scale)

                scale = scale.reshape(x_grouped.shape[0], x_grouped.shape[1])
                offset = offset.reshape(x_grouped.shape[0], x_grouped.shape[1])

                return scale.contiguous(), offset.contiguous()
            else:
                raise


@torch.no_grad()
def check_mse(w_fp, w_q, module_name=None):
    err = torch.mean(((w_fp.cpu() - w_q.cpu()) / (w_fp.cpu().std() + 1e-8))**2)
    if module_name is not None:
        print(module_name)
    print(f"error (w-wq)^2: {err.mean():.3f}")


class W8A8Linear(torch.nn.Module):
    def __init__(
            self, 
            weight_shape,
            bias,
            matmul_out_scale=1 / 1024.0
        ):
        super().__init__()
        self.weight_shape = weight_shape
        self.bias = bias
        assert self.bias == False

        self.input_quantizer = DynamicActivationQuantizer(
            quantization_mode="TOKEN_INT8_127",
            scale_dtype="float32"
        )

        self.register_buffer(
            "compressed_weight",
            torch.empty(
                self.weight_shape,
                dtype=torch.int8,
                requires_grad=False
            )
        )

        self.weight_scale = torch.nn.Parameter(
            torch.empty(
                self.weight_shape[0],
                requires_grad=True
            )
        )

        self.matmul_out_scale = torch.nn.Parameter(
            torch.tensor(
                matmul_out_scale, 
                dtype=torch.float32, 
                requires_grad=False
            )
        )


    @torch.no_grad()
    def configure(self, fp_module, device, verbose=False, *args, **kwargs):
        self.to(device)
        w_fp = fp_module.weight.data.to(device).float()
        
        weight_scale, _ = MinMaxInitializer()(
            x_grouped=(w_fp / self.matmul_out_scale).reshape(self.weight_shape[0], 1, self.weight_shape[1]),
            positive_clip=127,
            negative_clip=-128,
            use_offset=False
        )
        weight_scale = weight_scale[:, 0]
        self.weight_scale.data.copy_(weight_scale)

        w_scaled = (w_fp / self.matmul_out_scale) / weight_scale[:, None]
        w_compressed = torch.clamp(torch.round(w_scaled), -128, 127).to(torch.int8)
        self.compressed_weight.data.copy_(w_compressed)
        
        if verbose:
            w_q = w_compressed.float() * self.weight_scale[:, None] * self.matmul_out_scale
            check_mse(w_fp, w_q, kwargs.get('module_name', None))

       
    def forward(self, x):
        #x_dtype = x.dtype

        x_int8, x_scales = self.input_quantizer(x, simulation=False)
        x_scales = x_scales.to(torch.float16)
        w_int8 = self.compressed_weight
        w_scale = self.weight_scale

        # x = torch.nn.functional.linear(x_int8.float(), w_int8.float())
        
        x_shape = x.shape
        x = triton_matmul_int8_int8(x_int8.reshape(-1, x_int8.shape[-1]), w_int8.T).to(torch.float32)
        x = x.reshape(x_shape[0], x_shape[1], -1)
        x = (x * self.matmul_out_scale).to(torch.float16)


        # assert torch.isfinite(x).all()

        x = x * w_scale * x_scales[:, :, None]

        return x#.to(x_dtype)
