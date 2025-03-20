import torch

@torch.no_grad()
def pack_bool_tensor(x):
    """
    Packs a boolean tensor (2D or 3D) into a uint8 tensor.
    Args:
        x (torch.Tensor): Input boolean tensor of shape (H, W) or (C, H, W).
    Returns:
        packed (torch.Tensor): Packed uint8 tensor.
        original_shape (tuple): Original shape of the input tensor.
    """
    assert x.dtype == torch.bool, "Input tensor must be of type torch.bool"
    original_shape = x.shape

    # Flatten the tensor
    x_flat = x.flatten()

    # Pad to ensure length is a multiple of 8
    original_size = x_flat.numel()
    pad_size = (8 - (original_size % 8)) % 8
    x_padded = torch.cat([x_flat, torch.zeros(pad_size, dtype=torch.bool, device=x_flat.device)])

    # Pack 8 booleans into 1 byte (uint8)
    x_packed = x_padded.view(-1, 8)  # Reshape into groups of 8 bits
    bit_weights = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=x_flat.device)
    x_packed = (x_packed * bit_weights).sum(dim=1).to(torch.uint8)

    return x_packed, original_shape

@torch.no_grad()
def unpack_bool_tensor(packed, original_shape):
    """
    Unpacks a uint8 tensor into a boolean tensor of the original shape.
    Args:
        packed (torch.Tensor): Packed uint8 tensor.
        original_shape (tuple): Original shape of the boolean tensor.
    Returns:
        x (torch.Tensor): Unpacked boolean tensor.
    """
    # Unpack bytes to booleans
    bit_weights = torch.tensor(
        [1, 2, 4, 8, 16, 32, 64, 128], 
        dtype=torch.uint8,
        device=packed.device)
    x_padded = (packed.unsqueeze(-1).bitwise_and(bit_weights).ne(0)).view(-1)

    # Remove padding and reshape to original shape
    original_size = torch.prod(torch.tensor(original_shape)).item()
    x_original = x_padded[:original_size].reshape(original_shape)

    return x_original
