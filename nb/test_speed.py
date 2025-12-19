import os
DEVICES = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = DEVICES

from typing import Tuple
import time
import torch
# from modeling_quant_llama import QuantizedLlamaForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


def generate_and_count_tokens(
    model,
    tokenizer,
    batch_size: int,
    max_new_tokens: int,
    temperature: float = None,
) -> Tuple[int, float]:

    device = model.get_decoder().embed_tokens.weight.device

    # For this test I set empty prompt (padding not used)
    prompts = [""] * batch_size
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

    input_ids = inputs["input_ids"]
    input_len = input_ids.shape[1]

    gen_cfg = model.generation_config

    gen_cfg.max_new_tokens = max_new_tokens

    if temperature is not None:
        gen_cfg.temperature = temperature
    temperature = gen_cfg.temperature

    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            generation_config=gen_cfg,
            pad_token_id=tokenizer.pad_token_id,
            temperature=temperature
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # outputs shape: (batch_size, seq_len_generated = input_len + new_tokens)
    total_generated_tokens = 0
    for seq in outputs:
        seq_len = seq.shape[0]
        new_tokens = max(0, seq_len - input_len)
        total_generated_tokens += new_tokens

    return total_generated_tokens, elapsed


def benchmark(
    model,
    tokenizer,
    batch_size: int,
    max_new_tokens: int,
    temperature: float = None,
    n_warmup_runs=1,
    measured_runs=10,
):
    print("Warming up...")
    for i in range(n_warmup_runs):
        total_tokens, elapsed = generate_and_count_tokens(
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        print(f"  warmup #{i+1}: produced {total_tokens} tokens in {elapsed:.4f}s")
        print(f"  warmup speed: {total_tokens / elapsed} tok/s")

    print("Measured runs...")
    tok_per_sec_list = []
    per_run_details = []
    for i in range(measured_runs):
        total_tokens, elapsed = generate_and_count_tokens(
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        tps = total_tokens / elapsed if elapsed > 0 else float("inf")
        tok_per_sec_list.append(tps)
        per_run_details.append((i + 1, total_tokens, elapsed, tps))
        print(f"  run #{i+1}: {total_tokens} tokens, {elapsed:.4f}s -> {tps:,.1f} tok/s")

    mean_tps = torch.mean(torch.tensor(tok_per_sec_list))
    median_tps = torch.median(torch.tensor(tok_per_sec_list))
    std_tps = torch.std(torch.tensor(tok_per_sec_list))

    print("\n=== Summary ===")
    print(f"Batch size: {batch_size}; max_new_tokens: {max_new_tokens}; measured runs: {measured_runs}")
    print(f"Mean tok/s: {mean_tps:,.2f} +- {std_tps:,.2f}")
    print(f"Median tok/s: {median_tps:,.2f}")


# # Quantized model
# path2model = "./init_Llama-2-7B_w8a8"
# model = QuantizedLlamaForCausalLM.from_pretrained(path2model, device_map="cuda:0", dtype="auto")
# tokenizer = AutoTokenizer.from_pretrained(path2model)


# FP model
path2model = "/media/msst/ssd_storage1/ml/llm/pretrained_models/Llama2-7B"
model = AutoModelForCausalLM.from_pretrained(path2model, device_map="cuda:0", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(path2model) 


tokenizer.add_special_tokens({'pad_token': '[PAD]'})
benchmark(
    model,
    tokenizer,
    batch_size=1,
    max_new_tokens=1024,
    # temperature=2.0,
    n_warmup_runs=1,
    measured_runs=5,
)
