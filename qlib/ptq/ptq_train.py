@torch.no_grad()
def run_block_ptq(block, fp_activations, q_activations=None, collect_q_activations=True):
	if q_activations is None:
		q_activations = fp_activations
		print("set q_activations = fp_activations")

	# collect fp_activations
	switch_quantizers(block, 'fp')
	new_fp_activations = []
	for fp_batch in tqdm(fp_activations):
		fp_batch = fp_batch.to(DEVICE)
		fp_output = block(fp_batch)
		new_fp_activations.append(fp_output.detach().cpu())

	# collect q_activations
	switch_quantizers(block, 'quantize')
	if collect_q_activations:
		new_q_activations = []
		for q_batch in tqdm(q_activations):
			q_batch = q_batch.to(DEVICE)
			q_output = block(q_batch)
			new_q_activations.append(q_output.detach().cpu())
	else:
		new_q_activations = None

	return new_fp_activations, new_q_activations


def collect_optim_params(module):
	optim_params = []
	for param_name, param in module.named_parameters(recurse=True):
		if "weight_quantizer" in param_name:
			param.requires_grad = True
			optim_params.append(param)
		else:
			param.requires_grad = False
	return optim_params


def finetune_block_ptq(
		block, 
		fp_activations, 
		q_activations=None, 
		train_block=False):
	if q_activations is None:
		q_activations = fp_activations
		print("set q_activations = fp_activations")
		
	# collect fp activations
	switch_quantizers(block, 'fp')
	new_fp_activations = []
	with torch.no_grad():
		#for fp_batch in tqdm(fp_activations):
		for fp_batch in tqdm(q_activations):
			fp_batch = fp_batch.to(DEVICE)
			fp_input = prepare_llama_layer_inputs(
								activations=fp_batch,
								config = model.config,
								position_embeddings_func = model_decoder.rotary_emb
					)
			fp_output = block(**fp_input)[0]
			new_fp_activations.append(fp_output.detach().cpu())

	# collect q activations
	switch_quantizers(block, 'quantize')
	if train_block:
		with torch.enable_grad():
			loss_fn = qlib.MomentCriteria(p=2)
			optim = Adam(collect_optim_params(block), lr=1e-4)

			for step in tqdm(range(len(q_activations))):
				fp_output = new_fp_activations[step].to(DEVICE)
				q_batch = q_activations[step].to(DEVICE)
				
				q_input = prepare_llama_layer_inputs(
									activations=q_batch,
									config = model.config,
									position_embeddings_func = model_decoder.rotary_emb
						)
				
				q_output = block(**q_input)[0]
				loss = loss_fn(q_output, fp_output)
				loss.backward()
				#print(loss.item())
				optim.step()
				optim.zero_grad()

	new_q_activations = []
	with torch.no_grad():
		for q_batch in tqdm(q_activations):
			q_batch = q_batch.to(DEVICE)
			q_input = prepare_llama_layer_inputs(
								activations=q_batch,
								config = model.config,
								position_embeddings_func = model_decoder.rotary_emb
					)
			q_output = block(**q_input)[0]
			new_q_activations.append(q_output.detach().cpu())


	return new_fp_activations, new_q_activations



with torch.no_grad():
	with torch.autocast(device_type="cuda"):
		print("model_decoder.embed_tokens")

		fp_activations, q_activations = run_block_ptq(
			model_decoder.embed_tokens, fp_activations=train_ids
			)
		
		print("model_decoder.layers")

		for decoder_layer in model_decoder.layers:
			fp_activations, q_activations = finetune_block_ptq(
				decoder_layer, fp_activations, q_activations, train_block=True
				)
		
		# print("model_decoder.norm")

		# fp_activations, q_activations = run_block(
		# 	model_decoder.norm, fp_activations, q_activations, collect_q_activations=False
		# 	)

		# print("model.lm_head")

		# fp_activations, q_activations = run_block(
		# 	model.lm_head, fp_activations, q_activations, collect_q_activations=False
		# 	)

