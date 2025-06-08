export PYTHONPATH=/home/msst/repo/Quantization

lm_eval \
	--model hf \
    --model_args pretrained=/home/msst/repo/Quantization/ml/llm/pretrained_models/Llama2-7b-trellis \
    --tasks winogrande \
    --device cuda:0 \
    --batch_size 100 \
	--trust_remote_code
# ~2min