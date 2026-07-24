import os
os.environ["HF_HOME"]=os.getcwd()+"/pipecache"
import torch
from diffusers import AnimaModularPipeline
from safetensors.torch import (
	save_file,
	load_file
)
import math
import shutil
import json
import re
from lycoris import create_lycoris_from_weights
from lycoris.modules.locon import LoConModule
from lycoris.modules.loha import LohaModule
from lycoris.modules.lokr import LokrModule
from lycoris.modules.full import FullModule
from lycoris.modules.norms import NormModule
from lycoris.modules.diag_oft import DiagOFTModule
from lycoris.modules.boft import ButterflyOFTModule
from lycoris.modules.glora import GLoRAModule
from lycoris.modules.dylora import DyLoraModule
from lycoris.modules.ia3 import IA3Module
import gc
from huggingface_hub import snapshot_download

MODULE_LIST = [
	LoConModule,
	LohaModule,
	IA3Module,
	LokrModule,
	FullModule,
	NormModule,
	DiagOFTModule,
	ButterflyOFTModule,
	GLoRAModule,
	DyLoraModule,
]

def save_ckpt(keys,path):
	out_dict={}
	out_dict["__metadata__"]={"format":"pt"}
	n=0
	for k in keys:
		f=open(os.getcwd()+"/safe_temp/"+k+".safetensors","rb")
		l=int.from_bytes(f.read(8),byteorder="little")
		head=f.read(l).decode()
		head=json.loads(head)
		out_dict[k]=head[k]
		offsets=out_dict[k]["data_offsets"][1]
		out_dict[k]["data_offsets"][0]=n
		n=n+offsets
		out_dict[k]["data_offsets"][1]=n
		f.close()

	output=open(path,"wb")
	out_dict=str(out_dict).replace("'",'"')
	out_dict=out_dict.encode()
	l=len(out_dict).to_bytes(8,byteorder="little")
	output.write(l)
	output.write(out_dict)

	for k in keys:
		f=open(os.getcwd()+"/safe_temp/"+k+".safetensors","rb")
		l=int.from_bytes(f.read(8),byteorder="little")
		head=f.read(l)
		output.write(f.read())
		f.close()
		os.remove(os.getcwd()+"/safe_temp/"+k+".safetensors")
	output.close()

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
	
def zip_ckpt(pipe,transformer_sd,text_conditioner_sd,text_encoder_sd,vae_sd,full_file):
	block_maps_swap = {v: k for k, v in block_maps.items()}
	root_map_swap = {v: k for k, v in root_map.items()}
	keys=[]
	for k,p in pipe.transformer.named_parameters():
		t=p.data.to(torch.float32)
		if k in transformer_sd:
			sum1=torch.sum(torch.abs(t)).item()
			sum2=torch.sum(torch.abs(transformer_sd.pop(k).to(torch.float32))).item()
			n=not(math.isnan(sum1) or math.isnan(sum2))
			if n and sum1!=sum2:
				t=t*sum2/sum1

		block_re = re.compile(r"^transformer_blocks\.(\d+)\.(.+)$")
		m = block_re.match(k)
		if m==None:
			for k2 in root_map_swap:
				k=k.replace(k2,root_map_swap[k2])
			k="model.diffusion_model."+k
		else:
			block_index = m.group(1)
			tail = m.group(2)
			mapped_tail = block_maps_swap.get(tail)
			k="model.diffusion_model.blocks."+str(block_index)+"."+mapped_tail

		osd={}
		osd[k]=t.to(torch.bfloat16)
		save_file(osd,os.getcwd()+"/safe_temp/"+k+".safetensors")
		keys.append(k)
	del transformer_sd

	for k,p in pipe.text_conditioner.named_parameters():
		t=p.data.to(torch.float32)
		if k in text_conditioner_sd:
			sum1=torch.sum(torch.abs(t)).item()
			sum2=torch.sum(torch.abs(text_conditioner_sd.pop(k).to(torch.float32))).item()
			n=not(math.isnan(sum1) or math.isnan(sum2))
			if n and sum1!=sum2:
				t=t*sum2/sum1

		k="model.diffusion_model.llm_adapter."+k

		osd={}
		osd[k]=t.to(torch.bfloat16)
		save_file(osd,os.getcwd()+"/safe_temp/"+k+".safetensors")
		keys.append(k)
	del text_conditioner_sd

	if full_file:
		vae_keys1_swap={v: k for k, v in vae_keys1.items()}
		vae_keys2_swap={v: k for k, v in vae_keys2.items()}
		vae_keys3_swap={v: k for k, v in vae_keys3.items()}
		for k,p in pipe.vae.named_parameters():
			t=p.data.to(torch.float32)
			if k in vae_sd:
				sum1=torch.sum(torch.abs(t)).item()
				sum2=torch.sum(torch.abs(vae_sd.pop(k).to(torch.float32))).item()
				n=not(math.isnan(sum1) or math.isnan(sum2))
				if n and sum1!=sum2:
					t=t*sum2/sum1
			osd={}
			if k.startswith("encoder."):
				for k2 in vae_keys2_swap:
					k=k.replace(k2,vae_keys2_swap[k2])
			elif k.startswith("decoder."):
				for k2 in vae_keys3_swap:
					k=k.replace(k2,vae_keys3_swap[k2])
			else:
				for k2 in vae_keys1_swap:
					k=k.replace(k2,vae_keys1_swap[k2])
			k="first_stage_model."+k
			osd[k]=t.to(torch.bfloat16)
			save_file(osd,os.getcwd()+"/safe_temp/"+k+".safetensors")
			keys.append(k)
		del vae_sd
			
		for k,p in pipe.text_encoder.named_parameters():
			t=p.data.to(torch.float32)
			if k in text_encoder_sd:
				sum1=torch.sum(torch.abs(t)).item()
				sum2=torch.sum(torch.abs(text_encoder_sd.pop(k).to(torch.float32))).item()
				n=not(math.isnan(sum1) or math.isnan(sum2))
				if n and sum1!=sum2:
					t=t*sum2/sum1
			osd={}
			k="cond_stage_model.qwen3_06b.transformer.model."+k
			osd[k]=t.to(torch.bfloat16)
			save_file(osd,os.getcwd()+"/safe_temp/"+k+".safetensors")
			keys.append(k)
		del text_encoder_sd
	else:
		del vae_sd,text_encoder_sd
			
	return keys

def mksafe(base_path,loras,ws,out_path,full_file,win=None):
	if win!=None:
		win["RUN"].Update(disabled=True)

	if os.path.exists(os.getcwd()+"/safe_temp"):
		shutil.rmtree(os.getcwd()+"/safe_temp")
		
	if win!=None:
		win["info"].update("make pipe")
	else:
		print("make pipe")
	os.mkdir(os.getcwd()+"/safe_temp")

	try:
		if base_path.endswith(".safetensors"):
			transformer_sd,text_conditioner_sd,text_encoder_sd,vae_sd=safe2diff(safe_path=base_path,ff=full_file)
		else:
			transformer_sd,text_conditioner_sd,text_encoder_sd,vae_sd=folder2diff(path=base_path,ff=full_file)
		pipe = AnimaModularPipeline.from_pretrained(os.getcwd()+"/AnimaBaseV1")
		pipe.load_components()
		if transformer_sd!={}:
			for k,p in pipe.transformer.named_parameters():
				p.data=transformer_sd.pop(k)
		del transformer_sd
		if text_encoder_sd!={}:
			for k,p in pipe.text_encoder.named_parameters():
				p.data=text_encoder_sd.pop(k)
		del text_encoder_sd
		if text_conditioner_sd!={}:
			for k,p in pipe.text_conditioner.named_parameters():
				p.data=text_conditioner_sd.pop(k)
		del text_conditioner_sd
		if vae_sd!={}:
			for k,p in pipe.vae.named_parameters():
				p.data=vae_sd.pop(k)
		del vae_sd
	except:
		if win!=None:
			win["RUN"].Update(disabled=False)
			win["info"].update("Unsupported Anima checkpoint")
		else:
			print("Unsupported Anima checkpoint")
		shutil.rmtree(os.getcwd()+"/safe_temp")
		return
	gc.collect()

	if win!=None:
		win["info"].update("merge lora")
	else:
		print("merge lora")
	try:
		for i in range(len(loras)):
			sd=load_file(loras[i])
			MODULE_type=None
			for m in MODULE_LIST:
				for k in m.weight_list_det:
					for k2 in sd:
						if k2.endswith(k):
							MODULE_type=m
							break
						elif k2.endswith("lora_B.weight"):
							MODULE_type="B"
							break
					if MODULE_type!=None:
						break
				if MODULE_type!=None:
					break
			if MODULE_type==None:
				if win!=None:
					win["RUN"].Update(disabled=False)
					win["info"].update(loras[i]+" isn't supported.")
				else:
					print(loras[i]+" isn't supported.")
				return
			if MODULE_type=="B":
				MODULE_type=LoConModule
				key_dict=list(sd)
				for k2 in key_dict:
					if k2.endswith("lora_B.weight"):
						k=k2.replace("lora_B.weight","lora_up.weight")
						sd[k]=sd.pop(k2)
						kk=k2.replace("lora_B.weight","alpha")
						if not(kk in sd):
							sd[kk]=torch.tensor(sd[k].size()[1])
					elif k2.endswith("lora_A.weight"):
						k=k2.replace("lora_A.weight","lora_down.weight")
						sd[k]=sd.pop(k2)
			key_dict={}
			for k in sd:
				for k2 in MODULE_type.weight_list_det:
					if k.endswith("."+k2):
						k=k.removesuffix("."+k2)
						key_dict[k]=k.replace(".","_")

			transformer_sd={}
			text_encoder_sd={}
			text_conditioner_sd={}

			for k in key_dict:
				if key_dict[k].startswith("lora_te_"):
					key_dict[k]=key_dict[k].removeprefix("lora_te_")
					for k2 in MODULE_type.weight_list:
						if k+"."+k2 in sd:
							text_encoder_sd["lycoris_"+key_dict[k]+"."+k2]=sd.pop(k+"."+k2)
					continue
				elif key_dict[k].startswith("text_encoders_qwen3_06b_transformer_model_"):
					key_dict[k]=key_dict[k].removeprefix("text_encoders_qwen3_06b_transformer_model_")
					for k2 in MODULE_type.weight_list:
						if k+"."+k2 in sd:
							text_encoder_sd["lycoris_"+key_dict[k]+"."+k2]=sd.pop(k+"."+k2)
					continue

				m=re.search(r"llm_adapter_",key_dict[k])
				if m!=None:
					key_dict[k]=key_dict[k][m.end():]
					for k2 in MODULE_type.weight_list:
						if k+"."+k2 in sd:
							text_conditioner_sd["lycoris_"+key_dict[k]+"."+k2]=sd.pop(k+"."+k2)
						continue

				m=re.search(r"blocks_[0-9]+_",key_dict[k])
				if m!=None:
					key_dict[k]=key_dict[k][m.start():]
					for k2 in block_maps:
						mk2=k2.removesuffix(".weight").replace(".","_")
						mk2_value=block_maps[k2].removesuffix(".weight").replace(".","_")
						key_dict[k]=key_dict[k].replace(mk2,mk2_value)
					for k2 in MODULE_type.weight_list:
						if k+"."+k2 in sd:
							transformer_sd["lycoris_transformer_"+key_dict[k]+"."+k2]=sd.pop(k+"."+k2)
					continue

				for k3 in root_map:
					mk2=k3.removesuffix(".weight").replace(".","_")
					mk2_value=root_map[k3].removesuffix(".weight").replace(".","_")
					m=re.search(mk2,key_dict[k])
					if m!=None:
						key_dict[k]=mk2_value
						for k2 in MODULE_type.weight_list:
							if k+"."+k2 in sd:
								transformer_sd["lycoris_"+key_dict[k]+"."+k2]=sd.pop(k+"."+k2)
						break
					m=re.search(mk2_value,key_dict[k])
					if m!=None:
						key_dict[k]=mk2_value
						for k2 in MODULE_type.weight_list:
							if k+"."+k2 in sd:
								transformer_sd["lycoris_"+key_dict[k]+"."+k2]=sd.pop(k+"."+k2)
						break

			if transformer_sd=={} and text_encoder_sd=={} and text_conditioner_sd=={}:
				if win!=None:
					win["RUN"].Update(disabled=False)
					win["info"].update(loras[i]+" isn't supported.")
				else:
					print(loras[i]+" isn't supported.")
				shutil.rmtree(os.getcwd()+"/safe_temp")
				return
			if transformer_sd!={}:
				wrapper, _ = create_lycoris_from_weights(multiplier=ws[i],file="dummy.safetensors",module=pipe.transformer, weights_sd=transformer_sd)
				wrapper.merge_to()
				del wrapper
			del transformer_sd
			if text_encoder_sd!={}:
				wrapper, _ = create_lycoris_from_weights(multiplier=ws[i],file="dummy.safetensors",module=pipe.text_encoder, weights_sd=text_encoder_sd)
				wrapper.merge_to()
				del wrapper
			del text_encoder_sd
			if text_conditioner_sd!={}:
				wrapper, _ = create_lycoris_from_weights(multiplier=ws[i],file="dummy.safetensors",module=pipe.text_conditioner, weights_sd=text_conditioner_sd)
				wrapper.merge_to()
				del wrapper
			del text_conditioner_sd
	except:
		if win!=None:
			win["RUN"].Update(disabled=False)
			win["info"].update("I failed loading lora.")
		else:
			print("I failed loading lora.")
		shutil.rmtree(os.getcwd()+"/safe_temp")
		return

	if win!=None:
		win["info"].update("output ckpt file")
	else:
		print("output ckpt file")
	if base_path.endswith(".safetensors"):
		transformer_sd,text_conditioner_sd,text_encoder_sd,vae_sd=safe2diff(safe_path=base_path,ff=full_file)
	else:
		transformer_sd,text_conditioner_sd,text_encoder_sd,vae_sd=folder2diff(path=base_path,ff=full_file)
	keys=zip_ckpt(pipe,transformer_sd,text_conditioner_sd,text_encoder_sd,vae_sd,full_file)
	save_ckpt(keys,out_path)
	shutil.rmtree(os.getcwd()+"/safe_temp")

	if win!=None:
		win["RUN"].Update(disabled=False)
		win["info"].update("fin")
	else:
		print("fin")

if __name__=="__main__":
	import FreeSimpleGUI as sg
	import tkinter as tk
	import threading,pyperclip

	keys=[
		'ckpt','lora1','lora2','lora3',"out",'w1','w2','w3'
	]
	grp_rclick_menu={}
	for key in keys:
		grp_rclick_menu[key]=[
			"",
			[
				"-copy-::"+key,"-cut-::"+key,"-paste-::"+key
			]
		]
	layout =[
		[sg.Text("checkpoint file")],
		[
			sg.InputText(key='ckpt',right_click_menu=grp_rclick_menu["ckpt"]),
		],
		[
			sg.FileBrowse('select ckpt file', file_types=(('ckpt file', '.safetensors'),),key="file_browse", enable_events=True),
			sg.FolderBrowse("select ckpt folder",key="folder_browse", enable_events=True)
		],
		[sg.Text("lora1 file")],
		[sg.InputText(key='lora1',right_click_menu=grp_rclick_menu["lora1"]),sg.FileBrowse('select lora', file_types=(('lora file', '.safetensors'),))],
		[sg.Text("weight"),sg.InputText("1.0",key='w1',right_click_menu=grp_rclick_menu["w1"])],
		[sg.Text("lora2 file")],
		[sg.InputText(key='lora2',right_click_menu=grp_rclick_menu["lora2"]),sg.FileBrowse('select lora', file_types=(('lora file', '.safetensors'),))],
		[sg.Text("weight"),sg.InputText("1.0",key='w2',right_click_menu=grp_rclick_menu["w2"])],
		[sg.Text("lora3 file")],
		[sg.InputText(key='lora3',right_click_menu=grp_rclick_menu["lora3"]),sg.FileBrowse('select lora', file_types=(('lora file', '.safetensors'),))],
		[sg.Text("weight"),sg.InputText("1.0",key='w3',right_click_menu=grp_rclick_menu["w3"])],
		[sg.Checkbox('full file', key='ff')],
		[sg.Text("out file")],
		[sg.Input(key="out",right_click_menu=grp_rclick_menu["out"]),sg.FileSaveAs(file_types=(('ckpt file', '.safetensors'),))],
		[sg.Text("infomation",key="info")],
		[sg.Button('RUN'),sg.Button('EXIT')]
	]

	window = sg.Window('make safetensors anima', layout)

	while True:
		event, values = window.read()
		if event in (sg.WIN_CLOSED, 'EXIT'):
			break

		elif event=="RUN":
			base_safe=values["ckpt"]
			loras=[]
			weights=[]
			out_safe=values["out"]
			if values["lora1"]!="":
				loras.append(values["lora1"])
				try:
					weights.append(float(values["w1"]))
					window["w1"].update(str(float(values["w1"])))
				except:
					weights.append(1.0)
					window["w1"].update("1.0")
					
			if values["lora2"]!="":
				loras.append(values["lora2"])
				try:
					weights.append(float(values["w2"]))
					window["w2"].update(str(float(values["w2"])))
				except:
					weights.append(1.0)
					window["w2"].update("1.0")
					
			if values["lora3"]!="":
				loras.append(values["lora3"])
				try:
					weights.append(float(values["w3"]))
					window["w3"].update(str(float(values["w3"])))
				except:
					weights.append(1.0)
					window["w3"].update("1.0")
			full_file=values["ff"]

			if base_safe!="" and out_safe!="":
				thread1 = threading.Thread(target=mksafe,args=(base_safe,loras,weights,out_safe,full_file,window))
				thread1.start()
					
		elif "-copy-" in event:
			try:
				key=event.replace("-copy-::","")
				selected = window[key].widget.selection_get()
				pyperclip.copy(selected)
			except:
				pass
		elif "-cut-" in event:
			try:
				key=event.replace("-cut-::","")
				selected = window[key].widget.selection_get()
				pyperclip.copy(selected)
				window[key].widget.delete(tk.SEL_FIRST, tk.SEL_LAST)
			except:
				pass
		elif "-paste-" in event:
			try:
				key=event.replace("-paste-::","")
				selected = pyperclip.paste()
				insert_pos = window[key].widget.index("insert")
				window[key].Widget.insert(insert_pos, selected)
				window[key].widget.delete(tk.SEL_FIRST, tk.SEL_LAST)
			except:
				pass
		elif event=="folder_browse":
			window["ckpt"].update(values["folder_browse"])
		elif event=="file_browse":
			window["ckpt"].update(values["file_browse"])
				
	window.close()
