export PYTHONPATH=/home/msst/repo/Quantization

lm_eval \
	--model hf \
    --model_args pretrained=/home/msst/repo/Quantization/ml/llm/pretrained_models/Llama2-7b-trellis \
    --output_path /home/msst/repo/Quantization/logs/QA_results \
    --tasks piqa,winogrande,arc_easy,arc_challenge \
    --device cuda:0 \
    --batch_size 100 \
	--trust_remote_code 

# ~14.5min

# --log_samples  logging model responses 
