import gc
import torch
import safetensors
from safetensors.torch import (
	load_file,
	save_file
)
import os
import shutil
from huggingface_hub import snapshot_download
import re
import json

def flush():
	gc.collect()
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
	if torch.backends.mps.is_available():
		torch.mps.empty_cache()
	if torch.xpu.is_available():
		torch.xpu.empty_cache()

def getid(path,w=None):
	try:
		f=safetensors.safe_open(path, framework="pt", device="cpu")
		meta_dict=f.metadata()
	except:
		meta_dict={}

	if "id" in meta_dict:
		meta_id=meta_dict["id"]
		if "," in meta_id:
			meta_id=meta_id.split(",")
			for i in range(len(meta_id)):
				try:
					meta_id[i]=int(meta_id[i])
				except:
					meta_id[i]=""
		else:
			try:
				meta_id=[int(meta_id)]
			except:
				meta_id=[""]
	else:
		meta_id=[""]

	if w==None:
		if type(meta_id)==list:
			meta_id=str(meta_id[0])
		return meta_id

	if "weight" in meta_dict:
		meta_weight=meta_dict["weight"]
		if "," in meta_weight:
			meta_weight=meta_weight.split(",")
			for i in range(len(meta_weight)):
				try:
					meta_weight[i]=float(meta_weight[i])*w
				except:
					meta_weight[i]=w
		else:
			try:
				meta_weight=[float(meta_weight)*w]
			except:
				meta_weight=[w]
	else:
		meta_weight=[w]

	while True:
		if len(meta_weight)>=len(meta_id):
			break
		meta_weight.appned(w)

	meta_id2=[]
	meta_weight2=[]
	for i in range(len(meta_id)):
		if meta_id[i]!="":
			meta_id2.append(meta_id[i])
			meta_weight2.append(meta_weight[i])

	return meta_id2,meta_weight2

root_map = {
	"x_embedder.proj.1.weight": "patch_embed.proj.weight",
	"t_embedder.1.linear_1.weight": "time_embed.t_embedder.linear_1.weight",
	"t_embedder.1.linear_2.weight": "time_embed.t_embedder.linear_2.weight",
	"t_embedding_norm.weight": "time_embed.norm.weight",
	"final_layer.adaln_modulation.1.weight": "norm_out.linear_1.weight",
	"final_layer.adaln_modulation.2.weight": "norm_out.linear_2.weight",
	"final_layer.linear.weight": "proj_out.weight",
}

block_maps = {
	"adaln_modulation_self_attn.1.weight": "norm1.linear_1.weight",
	"adaln_modulation_self_attn.2.weight": "norm1.linear_2.weight",
	"adaln_modulation_cross_attn.1.weight": "norm2.linear_1.weight",
	"adaln_modulation_cross_attn.2.weight": "norm2.linear_2.weight",
	"adaln_modulation_mlp.1.weight": "norm3.linear_1.weight",
	"adaln_modulation_mlp.2.weight": "norm3.linear_2.weight",
	"self_attn.q_norm.weight": "attn1.norm_q.weight",
	"self_attn.k_norm.weight": "attn1.norm_k.weight",
	"self_attn.q_proj.weight": "attn1.to_q.weight",
	"self_attn.k_proj.weight": "attn1.to_k.weight",
	"self_attn.v_proj.weight": "attn1.to_v.weight",
	"self_attn.output_proj.weight": "attn1.to_out.0.weight",
	"cross_attn.q_norm.weight": "attn2.norm_q.weight",
	"cross_attn.k_norm.weight": "attn2.norm_k.weight",
	"cross_attn.q_proj.weight": "attn2.to_q.weight",
	"cross_attn.k_proj.weight": "attn2.to_k.weight",
	"cross_attn.v_proj.weight": "attn2.to_v.weight",
	"cross_attn.output_proj.weight": "attn2.to_out.0.weight",
	"mlp.layer1.weight": "ff.net.0.proj.weight",
	"mlp.layer2.weight": "ff.net.2.weight",
}

vae_keys1={
	"conv1":"quant_conv",
	"conv2":"post_quant_conv",
}
vae_keys2={
	"conv1":"conv_in",
	"head.0":"norm_out",
	"head.2":"conv_out",
	"downsamples":"down_blocks",
	"residual.2":"conv1",
	"residual.6":"conv2",
	"residual.0":"norm1",
	"residual.3":"norm2",
	"shortcut":"conv_shortcut",
	"middle.1":"mid_block.attentions.0",
	"middle.0":"mid_block.resnets.0",
	"middle.2":"mid_block.resnets.1",
}
vae_keys3={
	"conv1":"conv_in",
	"head.0":"norm_out",
	"head.2":"conv_out",
	"residual.2":"conv1",
	"residual.6":"conv2",
	"residual.0":"norm1",
	"residual.3":"norm2",
	"middle.1":"mid_block.attentions.0",
	"middle.0":"mid_block.resnets.0",
	"middle.2":"mid_block.resnets.1",
	"upsamples.3":"up_blocks.0.upsamplers.0",
	"upsamples.7":"up_blocks.1.upsamplers.0",
	"upsamples.11":"up_blocks.2.upsamplers.0",
	"upsamples.0":"up_blocks.0.resnets.0",
	"upsamples.10":"up_blocks.2.resnets.2",
	"upsamples.12":"up_blocks.3.resnets.0",
	"upsamples.13":"up_blocks.3.resnets.1",
	"upsamples.14":"up_blocks.3.resnets.2",
	"upsamples.1":"up_blocks.0.resnets.1",
	"upsamples.2":"up_blocks.0.resnets.2",
	"upsamples.4":"up_blocks.1.resnets.0",
	"shortcut":"conv_shortcut",
	"upsamples.5":"up_blocks.1.resnets.1",
	"upsamples.6":"up_blocks.1.resnets.2",
	"upsamples.8":"up_blocks.2.resnets.0",
	"upsamples.9":"up_blocks.2.resnets.1",
}

def safe2diff(safe_path,id=None):
	safe_path=safe_path.replace("\\","/")
	base_path=safe_path.removesuffix('.safetensors')
	if os.path.exists(base_path):
		shutil.rmtree(base_path)
	if not(os.path.exists(os.getcwd()+"/AnimaBaseV1")):
		snapshot_download(repo_id="circlestone-labs/Anima-Base-v1.0-Diffusers", local_dir=os.getcwd()+"/AnimaBaseV1")

		json_path=os.getcwd()+"/AnimaBaseV1/modular_model_index.json"
		f=open(json_path,"r")
		json_sd=json.load(f)
		f.close()

		json_sd["scheduler"][-1]["pretrained_model_name_or_path"]=os.getcwd()+"/AnimaBaseV1"
		json_sd["tokenizer"][-1]["pretrained_model_name_or_path"]=os.getcwd()+"/AnimaBaseV1"
		json_sd["t5_tokenizer"][-1]["pretrained_model_name_or_path"]=os.getcwd()+"/AnimaBaseV1"
		json_sd["text_conditioner"][-1]["pretrained_model_name_or_path"]=os.getcwd()+"/AnimaBaseV1"
		json_sd["text_encoder"][-1]["pretrained_model_name_or_path"]=os.getcwd()+"/AnimaBaseV1"
		json_sd["transformer"][-1]["pretrained_model_name_or_path"]=os.getcwd()+"/AnimaBaseV1"
		json_sd["vae"][-1]["pretrained_model_name_or_path"]=os.getcwd()+"/AnimaBaseV1"

		f=open(json_path,"w")
		json.dump(json_sd, f, indent=2)
		f.close()

	os.makedirs(base_path+"/text_conditioner")
	os.makedirs(base_path+"/text_encoder")
	os.makedirs(base_path+"/vae")
	os.makedirs(base_path+"/transformer")

	text_conditioner_path=base_path+"/text_conditioner/diffusion_pytorch_model.safetensors"
	transformer_path=base_path+"/transformer/diffusion_pytorch_model.safetensors"
	text_encoder_path=base_path+"/text_encoder/model.safetensors"
	vae_path=base_path+"/vae/diffusion_pytorch_model.safetensors"

	json_path=os.getcwd()+"/AnimaBaseV1/modular_model_index.json"
	f=open(json_path,"r")
	json_sd=json.load(f)
	f.close()

	json_sd["scheduler"][-1]["pretrained_model_name_or_path"]=os.getcwd()+"/AnimaBaseV1"
	json_sd["tokenizer"][-1]["pretrained_model_name_or_path"]=os.getcwd()+"/AnimaBaseV1"
	json_sd["t5_tokenizer"][-1]["pretrained_model_name_or_path"]=os.getcwd()+"/AnimaBaseV1"
	json_sd["text_conditioner"][-1]["pretrained_model_name_or_path"]=os.getcwd()+"/AnimaBaseV1"
	json_sd["text_encoder"][-1]["pretrained_model_name_or_path"]=os.getcwd()+"/AnimaBaseV1"
	json_sd["transformer"][-1]["pretrained_model_name_or_path"]=os.getcwd()+"/AnimaBaseV1"
	json_sd["vae"][-1]["pretrained_model_name_or_path"]=os.getcwd()+"/AnimaBaseV1"

	sd=load_file(safe_path)
	keys=[]
	head=None
	check="final_layer.linear.weight"
	pass_keys=[
		"pos_embedder.dim_spatial_range",
		"pos_embedder.dim_temporal_range",
		"pos_embedder.seq"
	]
	for k in sd:
		keys.append(k)
		if k.endswith(check):
			head=k.replace(check,"")

	text_conditioner_sd={}
	transformer_sd={}
	vae_sd={}
	text_encoder_sd={}		
	if head==None:
		raise RuntimeError("Unsupported Anima checkpoint")
	else:
		for k in keys:
			p=False
			for k2 in pass_keys:
				if k2 in k:
					p=True
			if p:
				continue

			if head=="":
				mk=k
			elif k.startswith(head):
				mk=k.removeprefix(head)
			elif k.startswith("first_stage_model."):
				mk=k.removeprefix("first_stage_model.")
				if k.startswith("first_stage_model.encoder."):
					for k2 in vae_keys2:
						mk=mk.replace(k2,vae_keys2[k2])
				elif k.startswith("first_stage_model.decoder."):
					for k2 in vae_keys3:
						mk=mk.replace(k2,vae_keys3[k2])
				else:
					for k2 in vae_keys1:
						mk=mk.replace(k2,vae_keys1[k2])
				vae_sd[mk]=sd[k]
				continue
			elif k.startswith("cond_stage_model.qwen3_06b.transformer.model."):
				mk=k.removeprefix("cond_stage_model.qwen3_06b.transformer.model.")
				text_encoder_sd[mk]=sd[k]
				continue
			else:
				continue

			if mk.startswith("llm_adapter"):
				mk=mk.removeprefix("llm_adapter.")
				text_conditioner_sd[mk]=sd[k]
				continue
			else:
				mapped = root_map.get(mk)
				if mapped is not None:
					transformer_sd[mapped] = sd[k]
					continue

				block_re = re.compile(r"^blocks\.(\d+)\.(.+)$")
				m = block_re.match(mk)
				if m is not None:
					block_index = m.group(1)
					tail = m.group(2)
					mapped_tail = block_maps.get(tail)
					if mapped_tail is None:
						raise RuntimeError(f"Unsupported Anima checkpoint key in blocks: {k}")
					transformer_sd[f"transformer_blocks.{block_index}.{mapped_tail}"] = sd[k]
					continue

			raise RuntimeError(f"Unsupported Anima checkpoint key: {k}")

	if transformer_sd!={}:
		save_file(transformer_sd,transformer_path)
		json_sd["transformer"][-1]["pretrained_model_name_or_path"]=base_path
		files = os.listdir(os.getcwd()+"/AnimaBaseV1/transformer")
		for file in files:
			if file!="diffusion_pytorch_model.safetensors":
				shutil.copy(
					os.getcwd()+"/AnimaBaseV1/transformer/"+file,
					base_path+"/transformer/"+file
				)
	if text_conditioner_sd!={}:
		save_file(text_conditioner_sd,text_conditioner_path)
		json_sd["text_conditioner"][-1]["pretrained_model_name_or_path"]=base_path
		files = os.listdir(os.getcwd()+"/AnimaBaseV1/text_conditioner")
		for file in files:
			if file!="diffusion_pytorch_model.safetensors":
				shutil.copy(
					os.getcwd()+"/AnimaBaseV1/text_conditioner/"+file,
					base_path+"/text_conditioner/"+file
				)
	if vae_sd!={}:
		save_file(vae_sd,vae_path)
		json_sd["vae"][-1]["pretrained_model_name_or_path"]=base_path
		files = os.listdir(os.getcwd()+"/AnimaBaseV1/vae")
		for file in files:
			if file!="diffusion_pytorch_model.safetensors":
				shutil.copy(
					os.getcwd()+"/AnimaBaseV1/vae/"+file,
					base_path+"/vae/"+file
				)
	if text_encoder_sd!={}:
		save_file(text_encoder_sd,text_encoder_path)
		json_sd["text_encoder"][-1]["pretrained_model_name_or_path"]=base_path
		files = os.listdir(os.getcwd()+"/AnimaBaseV1/text_encoder")
		for file in files:
			if file!="model.safetensors":
				shutil.copy(
					os.getcwd()+"/AnimaBaseV1/text_encoder/"+file,
					base_path+"/text_encoder/"+file
				)

	json_path=base_path+"/modular_model_index.json"
	f=open(json_path,"w")
	json.dump(json_sd, f, indent=2)
	f.close()

	if id!=None:
		f=open(base_path+"/id.txt","w")
		f.write(id)
		f.close()
	return base_path
