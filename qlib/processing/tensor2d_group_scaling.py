from typing import Union, Literal, Optional
import torch
# from qlib.processing.tensor_processing import TensorProcessing


class MinMaxInitializer:
    def __init__(self):
        pass

    @torch.no_grad()
    def __call__(self, x_grouped, use_offset, offset_mode, negative_clip, positive_clip):
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


class Tensor2dGroupScaling(torch.nn.Module):
    def __init__(
        self,
        tensor_shape,
        group_size: Union[int, Literal["tensor", "row"]],
        positive_clip: int,
        negative_clip: int,
        initialization_method: str = "minmax",
        use_offset: bool = False,
        offset_mode: Union[None, Literal["float", "midrize"]] = None,
        trainable: bool = True,
        use_scaled: bool = False,
    ):
        self.tensor_shape = tensor_shape
        self.positive_clip = positive_clip
        self.negative_clip = negative_clip
        self.trainable = trainable
        self.use_offset = use_offset

        # TODO: option "scaled" is not implemented yet
        assert use_scaled == False, "option 'scaled' is not implemented yet"
        self.use_scaled = use_scaled

        # TODO: option "trainable=False" is not implemented yet
        assert trainable == True, "option 'trainable=False' is not implemented yet"
        self.use_scaled = use_scaled

        valid_group_values = ("tensor", "row")  # , "col")
        if not (isinstance(group_size, int) or (group_size in valid_group_values)):
            raise AssertionError(f"group_size must int, 'tensor' or 'row', but got <{group_size}>!")
        self.group_size = group_size

        valid_initialization_methods = ("minmax",)
        if not (initialization_method in valid_initialization_methods):
            raise AssertionError(
                f"Only 'minmax' initialization methon supported now, but got <{initialization_method}>!"
            )
        self.initialization_method = initialization_method

        if self.use_offset:
            valid_offset_modes = ("float",)  # midrize
            if not (offset_mode in valid_offset_modes):
                raise AssertionError(f"Only 'float' offset mode supported now, but got <{offset_mode}>!")
            self.offset_mode = offset_mode
        else:
            self.offset_mode = None

        if self.group_size == "tensor":
            scale_shape = (1, 1)
        elif self.group_size == "row":
            scale_shape = (tensor_shape[0], 1)
        else:
            scale_shape = (tensor_shape[0], tensor_shape[1] // self.group_size)

        if self.trainable:
            self.scale = torch.nn.Parameter(torch.empty(scale_shape))
            if self.use_offset:
                self.offset = torch.nn.Parameter(torch.empty(scale_shape))

    def reshape_tensor_for_scaling(self, x):
        """
        Last dim - group size
        """
        assert x.shape == self.tensor_shape
        if self.group_size == "tensor":
            x_reshaped = x.reshape(1, 1, -1)
        elif self.group_size == "row":
            x_reshaped = x.reshape(x.shape[0], 1, x.shape[1])
        else:
            x_reshaped = x.reshape(x.shape[0], x.shape[1] // self.group_size, self.group_size)
        return x_reshaped

    @torch.no_grad()
    def configure(self, x):
        if self.initialization_method == "minmax":
            scale, offset = MinMaxInitializer()(
                x_grouped=self.reshape_tensor_for_scaling(x),
                use_offset=self.use_offset,
                offset_mode=self.offset_mode,
                negative_clip=self.negative_clip,
                positive_clip=self.positive_clip,
            )
        else:
            raise RuntimeError("Not Implemented Yet")

        if self.trainable:
            self.scale.copy_(scale)
            if self.use_offset:
                self.offset.copy_(offset)


    def forward(self, x):
        x = self.reshape_tensor_for_scaling(x)
        if self.use_offset:
            x = x + self.offset[:, :, None]
        x = x / self.scale[:, :, None]
        return x.reshape(self.tensor_shape)


    def inverse(self, x):
        x = self.reshape_tensor_for_scaling(x)
        x = x * self.scale[:, :, None]
        if self.use_offset:
            x = x - self.offset[:, :, None]
        return x.reshape(self.tensor_shape)