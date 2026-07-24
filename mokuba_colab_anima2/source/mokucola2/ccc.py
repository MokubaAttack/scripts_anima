import gc
import torch
import safetensors
from safetensors.torch import load_file
import os
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

def safe2diff(safe_path,ff):
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

		f=open(os.getcwd()+"/AnimaBaseV1/id.txt","w")
		f.write("2945208")
		f.close()

	sd=load_file(safe_path)
	keys=[]
	head=None
	check=("final_layer.linear.weight","proj_out.weight")

	for k in sd:
		keys.append(k)
		if k.endswith(check):
			head=k.removesuffix(check[0])
			head=head.removesuffix(check[1])

	text_conditioner_sd={}
	transformer_sd={}
	vae_sd={}
	text_encoder_sd={}
	ig_key=[]
	if head==None:
		raise RuntimeError("Unsupported Anima checkpoint")
	for k in keys:
		mk=k
		if k.startswith(head):
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

		if mk.startswith("llm_adapter"):
			mk=mk.removeprefix("llm_adapter.")
			text_conditioner_sd[mk]=sd[k]
			continue
			
		mapped = root_map.get(mk)
		if mapped is not None:
			transformer_sd[mapped] = sd[k]
			continue
		if mk in root_map.values():
			transformer_sd[mk] = sd[k]
			continue

		block_re = re.compile(r"^blocks\.(\d+)\.(.+)$")
		m = block_re.match(mk)
		if m is not None:
			block_index = m.group(1)
			tail = m.group(2)
			mapped_tail = block_maps.get(tail)
			if tail in block_maps.values():
				mapped_tail=tail
			if mapped_tail is not None:
				transformer_sd[f"transformer_blocks.{block_index}.{mapped_tail}"] = sd[k]
				continue
		ig_key.append(k)
		
	plus_key=[[],[],[],[]]

	sd2=load_file(os.getcwd()+"/AnimaBaseV1/text_conditioner/diffusion_pytorch_model.safetensors")
	for k in sd2:
		if not(k in text_conditioner_sd):
			text_conditioner_sd[k]=sd2[k]
			plus_key[1].append("text_conditioner."+k)
	if len(plus_key[1])==len(list(sd2)):
		plus_key[1]=["text_conditioner.all"]
	keys=list(text_conditioner_sd)
	for k in keys:
		if not(k in sd2):
			del text_conditioner_sd[k]

	sd2=load_file(os.getcwd()+"/AnimaBaseV1/transformer/diffusion_pytorch_model.safetensors")
	for k in sd2:
		if not(k in transformer_sd):
			transformer_sd[k]=sd2[k]
			plus_key[0].append("transformer."+k)
	if len(plus_key[0])==len(list(sd2)):
		plus_key[0]=["transformer.all"]
	keys=list(transformer_sd)
	for k in keys:
		if not(k in sd2):
			del transformer_sd[k]

	if ff:
		sd2=load_file(os.getcwd()+"/AnimaBaseV1/vae/diffusion_pytorch_model.safetensors")
		for k in sd2:
			if not(k in vae_sd):
				vae_sd[k]=sd2[k]
				plus_key[3].append("vae."+k)
		if len(plus_key[3])==len(list(sd2)):
			plus_key[3]=["vae.all"]
		keys=list(vae_sd)
		for k in keys:
			if not(k in sd2):
				del vae_sd[k]

		sd2=load_file(os.getcwd()+"/AnimaBaseV1/text_encoder/model.safetensors")
		for k in sd2:
			if not(k in text_encoder_sd):
				text_encoder_sd[k]=sd2[k]
				plus_key[2].append("text_encoder."+k)
		if len(plus_key[2])==len(list(sd2)):
			plus_key[2]=["text_encoder.all"]
		keys=list(text_encoder_sd)
		for k in keys:
			if not(k in sd2):
				del text_encoder_sd[k]
	else:
		vae_sd={}
		text_encoder_sd={}
	
	del sd2
	
	f=open(safe_path+".txt","w")
	f.write("minus\n")
	for k in ig_key:
		f.write(k+"\n")
	f.write("plus\n")
	for ks in plus_key:
		for k in ks:
			f.write(k+"\n")
	f.close()
	
	return transformer_sd,text_conditioner_sd,text_encoder_sd,vae_sd

def folder2diff(path,ff):
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

		f=open(os.getcwd()+"/AnimaBaseV1/id.txt","w")
		f.write("2945208")
		f.close()
	f=open(path+"/modular_model_index.json","r")
	json_sd=json.load(f)
	f.close()

	text_conditioner_path=json_sd["text_conditioner"][-1]["pretrained_model_name_or_path"]+"/"+json_sd["text_conditioner"][-1]["subfolder"]+"/diffusion_pytorch_model.safetensors"
	text_encoder_path=json_sd["text_encoder"][-1]["pretrained_model_name_or_path"]+"/"+json_sd["text_encoder"][-1]["subfolder"]+"/model.safetensors"
	transformer_path=json_sd["transformer"][-1]["pretrained_model_name_or_path"]+"/"+json_sd["transformer"][-1]["subfolder"]+"/diffusion_pytorch_model.safetensors"
	vae_path=json_sd["vae"][-1]["pretrained_model_name_or_path"]+"/"+json_sd["vae"][-1]["subfolder"]+"/diffusion_pytorch_model.safetensors"
	
	plus_key=[[],[],[],[]]
	ig_key=[]
	
	sd1=load_file(transformer_path)
	sd22=load_file(os.getcwd()+"/AnimaBaseV1/transformer/diffusion_pytorch_model.safetensors")
	for k in sd22:
		if not(k in sd1):
			sd1[k]=sd22[k]
			plus_key[0].append("transformer."+k)
	if len(plus_key[0])==len(list(sd22)):
		plus_key[0]=["transformer.all"]
	keys=list(sd1)
	for k in keys:
		if not(k in sd22):
			del sd1[k]
			ig_key.append("transformer."+k)
			
	sd2=load_file(text_conditioner_path)
	sd22=load_file(os.getcwd()+"/AnimaBaseV1/text_conditioner/diffusion_pytorch_model.safetensors")
	for k in sd22:
		if not(k in sd2):
			sd2[k]=sd22[k]
			plus_key[1].append("text_conditioner."+k)
	if len(plus_key[1])==len(list(sd22)):
		plus_key[1]=["text_conditioner.all"]
	keys=list(sd2)
	for k in keys:
		if not(k in sd22):
			del sd2[k]
			ig_key.append("text_conditioner."+k)
	if ff:
		sd3=load_file(text_encoder_path)
		sd22=load_file(os.getcwd()+"/AnimaBaseV1/text_encoder/model.safetensors")
		for k in sd22:
			if not(k in sd3):
				sd3[k]=sd22[k]
				plus_key[2].append("text_encoder."+k)
		if len(plus_key[2])==len(list(sd22)):
			plus_key[2]=["text_encoder.all"]
		keys=list(sd3)
		for k in keys:
			if not(k in sd22):
				del sd3[k]
				ig_key.append("text_encoder."+k)
				
		sd4=load_file(vae_path)
		sd22=load_file(os.getcwd()+"/AnimaBaseV1/vae/diffusion_pytorch_model.safetensors")
		for k in sd22:
			if not(k in sd4):
				sd4[k]=sd22[k]
				plus_key[3].append("vae."+k)
		if len(plus_key[3])==len(list(sd22)):
			plus_key[3]=["vae.all"]
		keys=list(sd4)
		for k in keys:
			if not(k in sd22):
				del sd4[k]
				ig_key.append("vae."+k)
	else:
		sd3={}
		sd4={}
		
	f=open(path+".txt","w")
	f.write("minus\n")
	for k in ig_key:
		f.write(k+"\n")
	f.write("plus\n")
	for ks in plus_key:
		for k in ks:
			f.write(k+"\n")
	f.close()
	
	return sd1,sd2,sd3,sd4
