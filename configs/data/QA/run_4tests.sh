export PYTHONPATH=/home/msst/repo/Quantization

lm_eval \
	--model hf \
    --model_args pretrained=/home/msst/repo/Quantization/logs/Llama2-7B_w8a8 \
    --output_path /home/msst/repo/Quantization/logs/QA_results \
    --tasks winogrande \
    --device cuda:0 \
    --batch_size 128 \
	--trust_remote_code 

# ~14.5min

# piqa,winogrande,arc_easy,arc_challenge

# --log_samples  logging model responses 

# 2.14, 1.40
# 