import torch
import torch.nn as nn
from transformers.masking_utils import create_causal_mask
from tqdm import tqdm


class QuantLinear(nn.Module):
    def __init__(self, weight_shape, group_size, bits):
        super().__init__()
        self.bits = bits
        self.weight_shape = weight_shape
        self.group_size = group_size
        self.scale_size = [weight_shape[0], weight_shape[1] // group_size]
        self.scale = nn.Parameter(torch.empty(self.scale_size))
        self.offset = nn.Parameter(torch.empty(self.scale_size))
        self.register_buffer(
            "compressed_weight",
            torch.empty(
                weight_shape,
                dtype=torch.uint8,
                requires_grad=False
            )
        )


    def reshape_weight_for_scaling(self, w):
        return w.reshape(
            self.weight_shape[0], self.weight_shape[1] // self.group_size, -1
        )


    @torch.compile()
    def reconstruct_weight(self):
        w = self.reshape_weight_for_scaling(self.compressed_weight)
        w = w * self.scale[..., None] - self.offset[..., None]
        w = w.reshape(self.weight_shape)
        return w


    def forward(self, x):
        w = self.reconstruct_weight()
        return torch.nn.functional.linear(x, w.to(x.dtype))


class MinMaxInitializer:
    def __init__(self, clip_ratio=1.0):
        self.clip_ratio = clip_ratio

    @torch.no_grad()
    def __call__(self, x_grouped, negative_clip, positive_clip):
        x_min = x_grouped.min(axis=-1)[0].unsqueeze(-1).float()
        x_max = x_grouped.max(axis=-1)[0].unsqueeze(-1).float()

        offset = (x_max * negative_clip - x_min * positive_clip) / (positive_clip - negative_clip)
        scale = (x_max + offset) / positive_clip
        scale = torch.abs(scale) * self.clip_ratio

        scale = scale.reshape(x_grouped.shape[0], x_grouped.shape[1])
        offset = offset.reshape(x_grouped.shape[0], x_grouped.shape[1])

        return scale.contiguous(), offset.contiguous()


@torch.no_grad()
def configure_single_layer(qlayer, layer, Cinv=None):
    max_int_val = 2**qlayer.bits - 1

    orig_weight = layer.weight.data
    if Cinv is not None:
        orig_weight = orig_weight.float() @ Cinv.float().T

    orig_weight_reshaped = qlayer.reshape_weight_for_scaling(orig_weight)
    scale, offset = MinMaxInitializer()(orig_weight_reshaped, negative_clip=0, positive_clip=max_int_val)
    
    quant_weight = (orig_weight_reshaped + offset[..., None]) / scale[..., None]
    quant_weight = quant_weight.reshape_as(orig_weight)

    quant_weight = torch.clamp(torch.round(quant_weight), 0, max_int_val).to(torch.uint8)

    qlayer.compressed_weight.copy_(quant_weight)
    qlayer.scale.copy_(scale)
    qlayer.offset.copy_(offset)


@torch.no_grad()
def init_quant_model_base(qmodel, model):
    for qmodule_name, qmodule in qmodel.named_modules():
        if isinstance(qmodule, QuantLinear):
            orig_module = model.get_submodule(qmodule_name)
            configure_single_layer(qmodule, orig_module)

            err = torch.mean(((orig_module.weight.data.cpu() - qmodule.reconstruct_weight().cpu()) / (orig_module.weight.data.cpu().std() + 1e-8))**2)
            print(err, qmodule_name)


@torch.no_grad()
def prepare_hessian(activations):
    hidden_size = activations[0].shape[-1]
    H = torch.zeros(hidden_size, hidden_size).cuda()
    for act in activations:
        act = act.cuda().view(-1, act.shape[-1]).float() / hidden_size ** 0.5
        H += act.T @ act
    return H


@torch.no_grad()
def prepare_hessian_q(activations, activations_q):
    hidden_size = activations[0].shape[-1]
    H_q = torch.zeros(hidden_size, hidden_size).cuda()
    for act_id in range(len(activations)):
        act = activations[act_id].cuda().view(-1, hidden_size).float() / hidden_size ** 0.5
        act_q = activations_q[act_id].cuda().view(-1, hidden_size).float() / hidden_size ** 0.5
        H_q += act.T @ act_q
    return H_q


@torch.no_grad()
def prepare_C(H, Hq):
    I = torch.eye(H.shape[0]).cuda() * 0.0 # * 1e-4 * Hq.trace() / H.shape[0]
    C = torch.linalg.inv(H.double() + I).float() @ Hq
    return C


@torch.no_grad()
def prepare_Cinv(H, Hq):
    I = torch.eye(H.shape[0]).cuda() * 0.0 # * 1e-4 * Hq.trace() / H.shape[0]
    C_inv = torch.linalg.inv(Hq.double() + I).float() @ H
    return C_inv


@torch.compile()
def hessian_loss(layer_q, layer_fp, H, C=None):
    w = layer_fp.weight
    w_q = layer_q.reconstruct_weight()
    if C is not None:
        w_q = w_q @ C.T
    delta_w = w_q - w
    return torch.trace(delta_w @ H @ delta_w.T)


def optimize_quant_params(
        layer_q,
        layer_fp,
        H,
        C=None
    ):
    trainable_params = [layer_q.scale, layer_q.offset]    
    optim = torch.optim.Adam(trainable_params, lr=1e-3)
    n_steps = 100

    for i in range(n_steps):
        optim.zero_grad()
        loss = hessian_loss(layer_q, layer_fp, H, C)
        if i == 0:
            init_loss =  loss.item()
        loss.backward()
        optim.step()

    print(f"{init_loss} -> {loss}")


@torch.compile()
def cholesky_loss(layer_q, layer_fp, L, C=None):
    # print("cholesky_loss!")
    w = layer_fp.weight
    w_q = layer_q.reconstruct_weight()
    if C is not None:
        w_q = w_q @ C.T
    delta_w = w_q - w
    return torch.sum((delta_w @ L) ** 2)


def optimize_quant_params_cholesky(
        layer_q,
        layer_fp,
        L,
        C=None
    ):
    trainable_params = [layer_q.scale, layer_q.offset]    
    optim = torch.optim.Adam(trainable_params, lr=1e-3)
    n_steps = 100

    for i in range(n_steps):
        optim.zero_grad()
        loss = cholesky_loss(layer_q, layer_fp, L, C)
        if i == 0:
            init_loss =  loss.item()
        loss.backward()
        optim.step()

    print(f"{init_loss} -> {loss}")


def hessian_loss_ste(layer_q, layer_fp, H):
    max_int_val = 2**layer_q.bits - 1

    latent_weight_reshaped = layer_q.reshape_weight_for_scaling(layer_fp.weight + layer_q.weight_addition)
    latent_weight_scaled = (latent_weight_reshaped + layer_q.offset[..., None]) / layer_q.scale[..., None]

    quant_weight = torch.clamp(torch.round(latent_weight_scaled), 0, max_int_val).to(torch.uint8)
    quant_weight_ste = quant_weight + latent_weight_scaled

    layer_q.compressed_weight.copy_(quant_weight.reshape_as(layer_fp.weight))

    weight_reco = quant_weight_ste * layer_q.scale[..., None] - layer_q.offset[..., None]
    weight_reco = weight_reco.reshape_as(layer_fp.weight)

    delta_w = weight_reco - layer_fp.weight

    C = 1e-6
    return torch.trace(delta_w @ H @ delta_w.T) + C * torch.sum(layer_q.weight_addition ** 2)


def optimize_quant_params_ste(
        layer_q,
        layer_fp,
        H
    ):
    layer_q.weight_addition = nn.Parameter(torch.zeros_like(layer_fp.weight.data).float())

    # trainable_params = [layer_q.scale, layer_q.offset, layer_q.latent_weight]        
    # trainable_params = [layer_q.scale, layer_q.offset]
    trainable_params = [layer_q.weight_addition]        
    optim = torch.optim.Adam(trainable_params, lr=1e-4)
    n_steps = 100

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        for i in range(n_steps):
            optim.zero_grad()

            loss = hessian_loss_ste(layer_q, layer_fp, H)
            if i == 0:
                init_loss =  loss.item()
            loss.backward()
            optim.step()

        print(f"{init_loss} -> {loss}")

    del layer_q.weight_addition


def init_quant_block_hessian(
        block_q,
        block_fp,
        activations,
        causal_mask,
        position_embeddings,
        ):

    ##### Attention #####

    # Copy activations for the residual stream
    residual_activations = [x.clone() for x in activations]

    # Collect activations after input_layernorm
    with torch.no_grad():
        for act_id in range(len(activations)):
            activations[act_id] = block_fp.input_layernorm(activations[act_id].cuda()).cpu()

    # Initialize q,k,v-projs
    H = prepare_hessian(activations)
    block_q_attn = block_q.self_attn
    block_fp_attn = block_fp.self_attn
    for layer_name in ["q_proj", "k_proj", "v_proj"]:
        layer_q = getattr(block_q_attn, layer_name)
        layer_fp = getattr(block_fp_attn, layer_name)
        configure_single_layer(layer_q, layer_fp)
        optimize_quant_params(layer_q, layer_fp, H)

    # Collect attention-out activations
    with torch.no_grad():
        for act_id in range(len(activations)):
            activations[act_id] = block_fp_attn.compute_attention(
                hidden_states=activations[act_id].cuda(), 
                position_embeddings=position_embeddings,
                attention_mask=causal_mask
            )[0].cpu()

    # Initialize o_proj
    layer_q = block_q.self_attn.o_proj
    layer_fp = block_fp.self_attn.o_proj
    H = prepare_hessian(activations)
    configure_single_layer(layer_q, layer_fp)
    optimize_quant_params(layer_q, layer_fp, H)

    # Collect self_attn outs
    with torch.no_grad():
        for act_id in range(len(activations)):
            act = block_fp.self_attn.o_proj(activations[act_id].cuda())
            res_act = residual_activations[act_id].cuda()
            activations[act_id] = (act + res_act).cpu()

    ##### MLP #####

    # Copy activations for the residual stream
    residual_activations = [x.clone() for x in activations]

    # Collect activations after post_attention_layernorm
    with torch.no_grad():
        for act_id in range(len(activations)):
            activations[act_id] = block_fp.post_attention_layernorm(activations[act_id].cuda()).cpu()

    # Initialize gate_proj and up_proj
    H = prepare_hessian(activations)
    block_q_mlp = block_q.mlp
    block_fp_mlp = block_fp.mlp
    for layer_name in ["gate_proj", "up_proj"]:
        layer_q = getattr(block_q_mlp, layer_name)
        layer_fp = getattr(block_fp_mlp, layer_name)
        configure_single_layer(layer_q, layer_fp)
        optimize_quant_params(layer_q, layer_fp, H)

    # Collect internal mlp activations
    with torch.no_grad():
        for act_id in range(len(activations)):
            act = activations[act_id].cuda()
            activations[act_id] = (block_fp_mlp.act_fn(block_fp.mlp.gate_proj(act)) * block_fp.mlp.up_proj(act)).cpu()

    # Initialize down_proj
    layer_q = block_q.mlp.down_proj
    layer_fp = block_fp.mlp.down_proj
    H = prepare_hessian(activations)
    configure_single_layer(layer_q, layer_fp)
    optimize_quant_params(layer_q, layer_fp, H)

    # Collect mlp outs
    with torch.no_grad():
        for act_id in range(len(activations)):
            act = block_fp.mlp.down_proj(activations[act_id].cuda())
            res_act = residual_activations[act_id].cuda()
            activations[act_id] = (act + res_act).cpu()


def init_quant_block_hessian_advanced(
        block_q,
        block_fp,
        activations,
        activations_q,
        causal_mask,
        position_embeddings,
        init_with_C=False,
        use_cholesky=False
        ):

    #####################
    ##### Attention #####
    #####################

    # Copy activations for the residual stream
    residual_activations = [x.clone() for x in activations]
    residual_activations_q = [x.clone() for x in activations_q]

    # Collect activations after input_layernorm
    with torch.no_grad():
        for act_id in range(len(activations)):
            activations[act_id] = block_fp.input_layernorm(activations[act_id].cuda()).cpu()
            activations_q[act_id] = block_q.input_layernorm(activations_q[act_id].cuda()).cpu()

    # Initialize q,k,v-projs
    H = prepare_hessian(activations)
    L = torch.linalg.cholesky(H) if use_cholesky else None
    Hq = prepare_hessian_q(activations, activations_q)
    C = prepare_C(H, Hq)
    Cinv = prepare_Cinv(H, Hq) if init_with_C else None

    block_q_attn = block_q.self_attn
    block_fp_attn = block_fp.self_attn
    for layer_name in ["q_proj", "k_proj", "v_proj"]:
        layer_q = getattr(block_q_attn, layer_name)
        layer_fp = getattr(block_fp_attn, layer_name)
        configure_single_layer(layer_q, layer_fp, Cinv)
        if use_cholesky:
            optimize_quant_params_cholesky(layer_q, layer_fp, L, C)
        else:
            optimize_quant_params(layer_q, layer_fp, H, C)

    # Collect attention-out activations
    with torch.no_grad():
        for act_id in range(len(activations)):
            activations[act_id] = block_fp_attn.compute_attention(
                hidden_states=activations[act_id].cuda(), 
                position_embeddings=position_embeddings,
                attention_mask=causal_mask
            )[0].cpu()
            activations_q[act_id] = block_q_attn.compute_attention(
                hidden_states=activations_q[act_id].cuda(), 
                position_embeddings=position_embeddings,
                attention_mask=causal_mask
            )[0].cpu()

    # Initialize o_proj    
    H = prepare_hessian(activations)
    L = torch.linalg.cholesky(H) if use_cholesky else None
    Hq = prepare_hessian_q(activations, activations_q)
    C = prepare_C(H, Hq)
    Cinv = prepare_Cinv(H, Hq) if init_with_C else None

    layer_q = block_q.self_attn.o_proj
    layer_fp = block_fp.self_attn.o_proj
    configure_single_layer(layer_q, layer_fp, Cinv)
    if use_cholesky:
        optimize_quant_params_cholesky(layer_q, layer_fp, L, C)
    else:
        optimize_quant_params(layer_q, layer_fp, H, C)

    # Collect self_attn outs
    with torch.no_grad():
        for act_id in range(len(activations)):
            act = activations[act_id].cuda()
            res_act = residual_activations[act_id].cuda()
            activations[act_id] = (block_fp.self_attn.o_proj(act) + res_act).cpu()

            act_q = activations_q[act_id].cuda()
            res_act_q = residual_activations_q[act_id].cuda()
            activations_q[act_id] = (block_q.self_attn.o_proj(act_q) + res_act_q).cpu()

    ###############
    ##### MLP #####
    ###############

    # Copy activations for the residual stream
    residual_activations = [x.clone() for x in activations]
    residual_activations_q = [x.clone() for x in activations_q]

    # Collect activations after post_attention_layernorm
    with torch.no_grad():
        for act_id in range(len(activations)):
            activations[act_id] = block_fp.post_attention_layernorm(activations[act_id].cuda()).cpu()
            activations_q[act_id] = block_q.post_attention_layernorm(activations_q[act_id].cuda()).cpu()

    # Initialize gate_proj and up_proj
    H = prepare_hessian(activations)
    L = torch.linalg.cholesky(H) if use_cholesky else None
    Hq = prepare_hessian_q(activations, activations_q)
    C = prepare_C(H, Hq)
    Cinv = prepare_Cinv(H, Hq) if init_with_C else None

    block_q_mlp = block_q.mlp
    block_fp_mlp = block_fp.mlp
    for layer_name in ["gate_proj", "up_proj"]:
        layer_q = getattr(block_q_mlp, layer_name)
        layer_fp = getattr(block_fp_mlp, layer_name)
        configure_single_layer(layer_q, layer_fp, Cinv)
        if use_cholesky:
            optimize_quant_params_cholesky(layer_q, layer_fp, L, C)
        else:
            optimize_quant_params(layer_q, layer_fp, H, C)

    # Collect internal mlp activations
    with torch.no_grad():
        for act_id in range(len(activations)):
            act = activations[act_id].cuda()
            activations[act_id] = (block_fp_mlp.act_fn(block_fp.mlp.gate_proj(act)) * block_fp.mlp.up_proj(act)).cpu()

            act_q = activations_q[act_id].cuda()
            activations_q[act_id] = (block_q_mlp.act_fn(block_q.mlp.gate_proj(act_q)) * block_q.mlp.up_proj(act_q)).cpu()

    # Initialize down_proj
    H = prepare_hessian(activations)
    L = torch.linalg.cholesky(H) if use_cholesky else None
    Hq = prepare_hessian_q(activations, activations_q)
    C = prepare_C(H, Hq)
    Cinv = prepare_Cinv(H, Hq) if init_with_C else None

    layer_q = block_q.mlp.down_proj
    layer_fp = block_fp.mlp.down_proj

    configure_single_layer(layer_q, layer_fp, Cinv)
    if use_cholesky:
        optimize_quant_params_cholesky(layer_q, layer_fp, L, C)
    else:
        optimize_quant_params(layer_q, layer_fp, H, C)

    # Collect mlp outs
    with torch.no_grad():
        for act_id in range(len(activations)):
            act = activations[act_id].cuda()
            res_act = residual_activations[act_id].cuda()
            activations[act_id] = (block_fp.mlp.down_proj(act) + res_act).cpu()

            act_q = activations_q[act_id].cuda()
            res_act_q = residual_activations_q[act_id].cuda()
            activations_q[act_id] = (block_q.mlp.down_proj(act) + res_act_q).cpu()


def init_quant_model_hessian(model_q, model_fp, dataloader, advanced=False, init_blocks=None, use_cholesky=False):
    embed_tokens = model_fp.get_decoder().embed_tokens.cuda()
    embed_tokens_device = embed_tokens.weight.device

    _batch = next(iter(dataloader))
    _inputs_embeds = embed_tokens(_batch.to(embed_tokens_device))

    cache_position = torch.arange(_inputs_embeds.shape[1], device=_inputs_embeds.device)
    position_ids = cache_position.unsqueeze(0)
    causal_mask = create_causal_mask(
        config=model_fp.config,
        input_embeds=_inputs_embeds,
        attention_mask=None,
        cache_position=cache_position,
        past_key_values=None,
        position_ids=position_ids,
    )

    position_embeddings = model_fp.get_decoder().rotary_emb(_inputs_embeds, position_ids)

    # Prepare activations
    activations = []
    with torch.no_grad():
        for batch in dataloader:
            activations.append(embed_tokens(batch.to(embed_tokens_device)).cpu())
    activations_q = [a.clone() for a in activations]

    for decoder_layer_id in tqdm(range(len(model_q.get_decoder().layers))):
        if init_blocks is not None:
            if (decoder_layer_id + 1) > init_blocks:
                break

        block_q = model_q.get_decoder().layers[decoder_layer_id].cuda()
        block_fp = model_fp.get_decoder().layers[decoder_layer_id].cuda()

        if not advanced:
            init_quant_block_hessian(
                block_q,
                block_fp,
                activations,
                causal_mask,
                position_embeddings,
            )
        else:
            init_quant_block_hessian_advanced(
                block_q,
                block_fp,
                activations,
                activations_q,
                causal_mask,
                position_embeddings,
                init_with_C=False,
                use_cholesky=use_cholesky
            )

        block_q = block_q.cpu()
        block_fp = block_fp.cpu()
