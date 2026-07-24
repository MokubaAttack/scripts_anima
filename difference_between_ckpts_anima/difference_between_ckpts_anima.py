import os
os.environ["HF_HOME"]=os.getcwd()+"/pipecache"
import json
from huggingface_hub import snapshot_download
from safetensors.torch import (
	load_file,
	save_file
)
import shutil
import torch
import gc
import re

CLAMP_QUANTILE=0.99

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
block_maps_swap = {v: k.removesuffix(".weight").replace(".","_") for k, v in block_maps.items()}
root_map_swap = {v: k.removesuffix(".weight").replace(".","_") for k, v in root_map.items()}

def safe2diff(safe_path,transfomer_out,text_conditioner_out,text_encoder_out):
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
	text_encoder_sd={}
	ig_key=[]
	if head==None:
		raise RuntimeError("Unsupported Anima checkpoint")
	for k in keys:
		mk=k
		if k.startswith(head):
			mk=k.removeprefix(head)
		elif k.startswith("cond_stage_model.qwen3_06b.transformer.model."):
			mk=k.removeprefix("cond_stage_model.qwen3_06b.transformer.model.")
			if mk.startswith("layers"):
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

	if text_conditioner_out:
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
		del sd2
	else:
		text_conditioner_sd={}
		
	if transfomer_out:
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
		del sd2
	else:
		transformer_sd={}
	if text_encoder_out:
		for k in sd2:
			if not(k in text_encoder_sd) and k.startswith("layers"):
				text_encoder_sd[k]=sd2[k]
				plus_key[2].append("text_encoder."+k)
		if len(plus_key[2])==len(list(sd2)):
			plus_key[2]=["text_encoder.all"]
		keys=list(text_encoder_sd)
		for k in keys:
			if not(k in sd2):
				del text_encoder_sd[k]
		del sd2
	else:
		text_encoder_sd={}
		
	f=open(safe_path+".txt","w")
	f.write("minus\n")
	for k in ig_key:
		f.write(k+"\n")
	f.write("plus\n")
	for ks in plus_key:
		for k in ks:
			f.write(k+"\n")
	f.close()

	return transformer_sd,text_conditioner_sd,text_encoder_sd

def folder2diff(path,transfomer_out,text_conditioner_out,text_encoder_out):
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
	
	plus_key=[[],[],[],[]]
	ig_key=[]
	
	if transfomer_out:
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
	else:
		sd1={}
	if text_conditioner_out:
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
	else:
		sd2={}
	if text_encoder_out:
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
	else:
		sd3={}
		
	f=open(path+".txt","w")
	f.write("minus\n")
	for k in ig_key:
		f.write(k+"\n")
	f.write("plus\n")
	for ks in plus_key:
		for k in ks:
			f.write(k+"\n")
	f.close()
	return sd1,sd2,sd3

def main_part(
	paths=[],
	dim=4,
	transfomer_out=True,
	text_conditioner_out=True,
	text_encoder_out=True,
	out_path=None,
	win=None
):
	if win!=None:
		win["RUN"].Update(disabled=True)

	for path in paths:
		if not(os.path.exists(path)):
			if win==None:
				print(path+" does not exist.")
			else:
				win["RUN"].Update(disabled=False)
				win["info"].update(path+" does not exist.")
			return
			
	if transfomer_out==False and text_conditioner_out==False and text_encoder_out==False:
		if win==None:
			print("You choose no contents.")
		else:
			win["RUN"].Update(disabled=False)
			win["info"].update("You choose no contents.")
		return
		
	temp_path=os.getcwd()+"/safe_temp"
	if os.path.exists(temp_path):
		shutil.rmtree(temp_path)
	os.mkdir(temp_path)
	
	if win==None:
		print("load "+paths[0])
	else:
		win["info"].update("load "+paths[0])
	try:
		if paths[0].endswith("safetensors"):
			sd10,sd11,sd12=safe2diff(
				safe_path=paths[0],
				transfomer_out=transfomer_out,
				text_conditioner_out=text_conditioner_out,
				text_encoder_out=text_encoder_out
				)
		else:
			sd10,sd11,sd12=folder2diff(
				path=paths[0],
				transfomer_out=transfomer_out,
				text_conditioner_out=text_conditioner_out,
				text_encoder_out=text_encoder_out
				)
	except:
		if win==None:
			print("I couldn't load "+paths[0]+".")
		else:
			win["RUN"].Update(disabled=False)
			win["info"].update("I couldn't load "+paths[0]+".")
		shutil.rmtree(temp_path)
		return
		
	if win==None:
		print("load "+paths[1])
	else:
		win["info"].update("load "+paths[1])
	try:
		if paths[1].endswith("safetensors"):
			sd20,sd21,sd22=safe2diff(
				safe_path=paths[1],
				transfomer_out=transfomer_out,
				text_conditioner_out=text_conditioner_out,
				text_encoder_out=text_encoder_out
				)
		else:
			sd20,sd21,sd22=folder2diff(
				path=paths[1],
				transfomer_out=transfomer_out,
				text_conditioner_out=text_conditioner_out,
				text_encoder_out=text_encoder_out
				)
	except:
		if win==None:
			print("I couldn't load "+paths[1]+".")
		else:
			win["RUN"].Update(disabled=False)
			win["info"].update("I couldn't load "+paths[1]+".")
		shutil.rmtree(temp_path)
		return

	keys0=list(sd10)
	keys1=list(sd11)
	keys2=list(sd12)
	data_dict={}
	dict_sum=len(keys0+keys1+keys2)
	key_count=0

	for name in keys0:
		key_count=key_count+1
		if win!=None:
			win["info"].update("differ "+str(key_count)+"/"+str(dict_sum))
		else:
			print("\rdiffer "+str(key_count)+"/"+str(dict_sum),end="")

		try:
			t1=sd10.pop(name).to(torch.float)
			t2=sd20.pop(name).to(torch.float)
		except:
			if win==None:
				print("Unsupported Anima checkpoint key: "+name)
			else:
				win["RUN"].Update(disabled=False)
				win["info"].update("Unsupported Anima checkpoint key: "+name)
			shutil.rmtree(os.getcwd()+"/safe_temp")
			return
		mat=t2-t1

		if mat.dim()==1:
			continue
		if torch.any(torch.isnan(mat)):
			continue

		mapped = root_map_swap.get(name)
		if mapped==None:
			block_re = re.compile(r"^transformer_blocks\.(\d+)\.(.+)$")
			m = block_re.match(name)
			if m is not None:
				block_index = m.group(1)
				tail = m.group(2)
				mapped_tail = block_maps_swap.get(tail)
				if mapped_tail is None:
					if win==None:
						print("Unsupported Anima checkpoint key: "+name)
					else:
						win["RUN"].Update(disabled=False)
						win["info"].update("Unsupported Anima checkpoint key: "+name)
					shutil.rmtree(temp_path)
					return
				name="lora_unet_blocks_"+str(block_index)+"_"+str(mapped_tail)
			else:
				if win==None:
					print("Unsupported Anima checkpoint key: "+name)
				else:
					win["RUN"].Update(disabled=False)
					win["info"].update("Unsupported Anima checkpoint key: "+name)
				shutil.rmtree(temp_path)
				return
		else:
			name="lora_unet_"+mapped

		conv2d = len(mat.size()) == 4
		kernel_size = None if not conv2d else mat.size()[2:4]
		conv2d_3x3 = conv2d and kernel_size != (1, 1)
		out_dim, in_dim = mat.size()[0:2]

		if conv2d:
			if conv2d_3x3:
				mat = mat.flatten(start_dim=1)
			else:
				mat = mat.squeeze()

		module_new_rank = dim
		module_new_rank = min(module_new_rank, in_dim, out_dim)

		U, S, Vh = torch.linalg.svd(mat)

		U = U[:, :module_new_rank]
		S = S[:module_new_rank]
		U = U @ torch.diag(S)

		Vh = Vh[:module_new_rank, :]

		dist = torch.cat([U.flatten(), Vh.flatten()])
		hi_val = torch.quantile(dist, CLAMP_QUANTILE)
		low_val = -hi_val

		U = U.clamp(low_val, hi_val)
		Vh = Vh.clamp(low_val, hi_val)

		if conv2d:
			U = U.reshape(out_dim, module_new_rank, 1, 1)
			Vh = Vh.reshape(module_new_rank, in_dim, kernel_size[0], kernel_size[1])

		up_weight = U
		down_weight = Vh

		data_dict[name + ".lora_B.weight"] = up_weight.to(torch.bfloat16).contiguous()
		data_dict[name + ".lora_A.weight"] = down_weight.to(torch.bfloat16).contiguous()
		del t1,t2,mat,up_weight,down_weight,U,S,Vh,dist,hi_val,low_val
		gc.collect()
	del sd10,sd20

	for name in keys1:
		key_count=key_count+1
		if win!=None:
			win["info"].update("differ "+str(key_count)+"/"+str(dict_sum))
		else:
			print("\rdiffer "+str(key_count)+"/"+str(dict_sum),end="")

		try:
			t1=sd11.pop(name).to(torch.float)
			t2=sd21.pop(name).to(torch.float)
		except:
			if win==None:
				print("Unsupported Anima checkpoint key: "+name)
			else:
				win["RUN"].Update(disabled=False)
				win["info"].update("Unsupported Anima checkpoint key: "+name)
			shutil.rmtree(os.getcwd()+"/safe_temp")
			return
		mat=t2-t1

		if mat.dim()==1:
			continue
		if torch.any(torch.isnan(mat)):
			continue

		name="lora_unet_llm_adapter_"+name.removesuffix(".weight").replace(".","_")

		conv2d = len(mat.size()) == 4
		kernel_size = None if not conv2d else mat.size()[2:4]
		conv2d_3x3 = conv2d and kernel_size != (1, 1)
		out_dim, in_dim = mat.size()[0:2]

		if conv2d:
			if conv2d_3x3:
				mat = mat.flatten(start_dim=1)
			else:
				mat = mat.squeeze()

		module_new_rank = dim
		module_new_rank = min(module_new_rank, in_dim, out_dim)

		U, S, Vh = torch.linalg.svd(mat)

		U = U[:, :module_new_rank]
		S = S[:module_new_rank]
		U = U @ torch.diag(S)

		Vh = Vh[:module_new_rank, :]

		dist = torch.cat([U.flatten(), Vh.flatten()])
		hi_val = torch.quantile(dist, CLAMP_QUANTILE)
		low_val = -hi_val

		U = U.clamp(low_val, hi_val)
		Vh = Vh.clamp(low_val, hi_val)

		if conv2d:
			U = U.reshape(out_dim, module_new_rank, 1, 1)
			Vh = Vh.reshape(module_new_rank, in_dim, kernel_size[0], kernel_size[1])

		up_weight = U
		down_weight = Vh

		data_dict[name + ".lora_B.weight"] = up_weight.to(torch.bfloat16).contiguous()
		data_dict[name + ".lora_A.weight"] = down_weight.to(torch.bfloat16).contiguous()
		del t1,t2,mat,up_weight,down_weight,U,S,Vh,dist,hi_val,low_val
		gc.collect()
	del sd11,sd21

	for name in keys2:
		key_count=key_count+1
		if win!=None:
			win["info"].update("differ "+str(key_count)+"/"+str(dict_sum))
		else:
			print("\rdiffer "+str(key_count)+"/"+str(dict_sum),end="")

		try:
			t1=sd12.pop(name).to(torch.float)
			t2=sd22.pop(name).to(torch.float)
		except:
			if win==None:
				print("Unsupported Anima checkpoint key: "+name)
			else:
				win["RUN"].Update(disabled=False)
				win["info"].update("Unsupported Anima checkpoint key: "+name)
			shutil.rmtree(os.getcwd()+"/safe_temp")
			return

		mat=t2-t1

		if mat.dim()==1:
			continue
		if torch.any(torch.isnan(mat)):
			continue

		name="lora_te_"+name.removesuffix(".weight").replace(".","_")

		conv2d = len(mat.size()) == 4
		kernel_size = None if not conv2d else mat.size()[2:4]
		conv2d_3x3 = conv2d and kernel_size != (1, 1)
		out_dim, in_dim = mat.size()[0:2]

		if conv2d:
			if conv2d_3x3:
				mat = mat.flatten(start_dim=1)
			else:
				mat = mat.squeeze()

		module_new_rank = dim
		module_new_rank = min(module_new_rank, in_dim, out_dim)

		U, S, Vh = torch.linalg.svd(mat)

		U = U[:, :module_new_rank]
		S = S[:module_new_rank]
		U = U @ torch.diag(S)

		Vh = Vh[:module_new_rank, :]

		dist = torch.cat([U.flatten(), Vh.flatten()])
		hi_val = torch.quantile(dist, CLAMP_QUANTILE)
		low_val = -hi_val

		U = U.clamp(low_val, hi_val)
		Vh = Vh.clamp(low_val, hi_val)

		if conv2d:
			U = U.reshape(out_dim, module_new_rank, 1, 1)
			Vh = Vh.reshape(module_new_rank, in_dim, kernel_size[0], kernel_size[1])

		up_weight = U
		down_weight = Vh

		data_dict[name + ".lora_B.weight"] = up_weight.to(torch.bfloat16).contiguous()
		data_dict[name + ".lora_A.weight"] = down_weight.to(torch.bfloat16).contiguous()
		del t1,t2,mat,up_weight,down_weight,U,S,Vh,dist,hi_val,low_val
		gc.collect()
	del sd12,sd22

	save_file(data_dict,out_path)
	shutil.rmtree(temp_path)
	if win==None:
		print("")
		print("fin")
	else:
		win["RUN"].Update(disabled=False)
		win["info"].update("fin")
			
if __name__=="__main__":
	import tkinter as tk
	import pyperclip
	import threading
	import FreeSimpleGUI as sg

	sg.theme('GrayGrayGray')

	keys=["ckpt1","ckpt2","out","dim"]

	grp_rclick_menu={}
	for key in keys:
		grp_rclick_menu[key]=[
			"",
			[
				"-copy-::"+key,"-cut-::"+key,"-paste-::"+key
			]
		]

	layout=[
		[
			sg.Text("model_org"), sg.Input(key="ckpt1",right_click_menu=grp_rclick_menu["ckpt1"]),
			sg.FileBrowse('select ckpt file', file_types=(('ckpt file', '.safetensors'),),key="file_browse1", enable_events=True),
			sg.FolderBrowse("select ckpt folder",key="folder_browse1", enable_events=True)
		],
		[
			sg.Text("model_tuned"), sg.Input(key="ckpt2",right_click_menu=grp_rclick_menu["ckpt2"]),
			sg.FileBrowse('select ckpt file', file_types=(('ckpt file', '.safetensors'),),key="file_browse2", enable_events=True),
			sg.FolderBrowse("select ckpt folder",key="folder_browse2", enable_events=True)
		],
		[sg.Text("dim"), sg.Input(key="dim",right_click_menu=grp_rclick_menu["dim"])],
		[sg.Text("output path"), sg.Input(key="out",right_click_menu=grp_rclick_menu["out"]),sg.FileSaveAs(file_types=(('lora file', '.safetensors'),))],
		[sg.Checkbox('transformer', default=True, key='unet'),sg.Checkbox('text_conditioner', default=True, key='text2'),sg.Checkbox('text_encoder', default=True, key='text1')],
		[sg.Text("infomation",key="info")],
		[sg.Button('RUN', key='RUN'),sg.Button('EXIT', key='EXIT')]
	]

	window = sg.Window('extract lora from anima models', layout)

	while True:
		event, values = window.read()
		if event == sg.WINDOW_CLOSED:
			break
		elif event=="EXIT":
			break
		elif event=="RUN":
			if values["out"]!="" and values["ckpt1"]!="" and values["ckpt2"]!="":
				if values["dim"]=="":
					dim=4
				else:
					try:
						dim=abs(int(values["dim"]))
					except:
						dim=4
				window["dim"].update(str(dim))
				out_path=values["out"]
				paths=[
					values["ckpt1"],values["ckpt2"]
				]

				trans_out=values["unet"]
				text_out=values["text1"]
				text_conditioner_out=values["text2"]
				
				thread1 = threading.Thread(target=main_part,args=(paths,dim,trans_out,text_conditioner_out,text_out,out_path,window))
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
		elif event=="file_browse1":
			window["ckpt1"].update(values["file_browse1"])
		elif event=="folder_browse1":
			window["ckpt1"].update(values["folder_browse1"])
		elif event=="file_browse2":
			window["ckpt2"].update(values["file_browse2"])
		elif event=="folder_browse2":
			window["ckpt2"].update(values["folder_browse2"])
		
	window.close()
