import os
os.environ["HF_HOME"]=os.getcwd()+"/pipecache"
import torch
from diffusers_anima import AnimaPipeline
from safetensors.torch import (
	save_file,
	load_file
)
import math
import shutil
import json
from diffusers_anima.pipelines.anima.loading import (
	_strip_wrapping_prefixes,
	_vae_text_check
)
from diffusers_anima.models.transformers.modeling_anima_transformer import _convert_anima_state_dict_to_diffusers
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

trans_key1={
	"ff.net.0.proj.weight":"mlp.layer1.weight",
	"ff.net.2.weight":"mlp.layer2.weight",
	"norm1.linear_1.weight":"adaln_modulation_self_attn.1.weight",
	"norm1.linear_2.weight":"adaln_modulation_self_attn.2.weight",
	"attn1.norm_q.weight":"self_attn.q_norm.weight",
	"attn1.norm_k.weight":"self_attn.k_norm.weight",
	"attn1.to_q.weight":"self_attn.q_proj.weight",
	"attn1.to_k.weight":"self_attn.k_proj.weight",
	"attn1.to_v.weight":"self_attn.v_proj.weight",
	"attn1.to_out.0.weight":"self_attn.output_proj.weight",
	"norm2.linear_1.weight":"adaln_modulation_cross_attn.1.weight",
	"norm2.linear_2.weight":"adaln_modulation_cross_attn.2.weight",
	"attn2.norm_q.weight":"cross_attn.q_norm.weight",
	"attn2.norm_k.weight":"cross_attn.k_norm.weight",
	"attn2.to_q.weight":"cross_attn.q_proj.weight",
	"attn2.to_k.weight":"cross_attn.k_proj.weight",
	"attn2.to_v.weight":"cross_attn.v_proj.weight",
	"attn2.to_out.0.weight":"cross_attn.output_proj.weight",
	"norm3.linear_1.weight":"adaln_modulation_mlp.1.weight",
	"norm3.linear_2.weight":"adaln_modulation_mlp.2.weight"
}
trans_key2={
	"patch_embed.proj.weight":"x_embedder.proj.1.weight",
	"time_embed.t_embedder.linear_1.weight":"t_embedder.1.linear_1.weight",
	"time_embed.t_embedder.linear_2.weight":"t_embedder.1.linear_2.weight",
	"time_embed.norm.weight":"t_embedding_norm.weight",
	"norm_out.linear_1.weight":"final_layer.adaln_modulation.1.weight",
	"norm_out.linear_2.weight":"final_layer.adaln_modulation.2.weight",
	"proj_out.weight":"final_layer.linear.weight"
}

vae_key1={
	"conv1":"quant_conv",
	"conv2":"post_quant_conv",
}
vae_key2={
	"downsamples":"down_blocks",
	"residual.2":"conv1",
	"residual.6":"conv2",
	"residual.0":"norm1",
	"residual.3":"norm2",
	"shortcut":"conv_shortcut",
	"middle.1":"mid_block.attentions.0",
	"middle.0":"mid_block.resnets.0",
	"middle.2":"mid_block.resnets.1",
	"conv1":"conv_in",
	"head.0":"norm_out",
	"head.2":"conv_out",
}
vae_key3={
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
	"conv1":"conv_in",
	"head.0":"norm_out",
	"head.2":"conv_out",
}
	
def zip_ckpt(pipe,sd,full_file):
	keys=[]
	for k,p in getattr(pipe, "transformer").named_parameters():
		t=p.data.to(torch.float32)
		sum1=torch.sum(torch.abs(t)).item()
		sum2=torch.sum(torch.abs(sd[k].to(torch.float32))).item()
		n=not(math.isnan(sum1) or math.isnan(sum2))
		if n and sum1!=sum2:
			t=t*sum2/sum1
		osd={}
		if k.startswith("core."):
			k=k.replace("core.","")
			if k.startswith("transformer_"):
				k=k.replace("transformer_","")
				for key in trans_key1:
					if k.endswith(key):
						k=k.replace(key,trans_key1[key])
			else:
				k=trans_key2[k]
		k="model.diffusion_model."+k
		osd[k]=t.to(torch.bfloat16)
		save_file(osd,os.getcwd()+"/safe_temp/"+k+".safetensors")
		keys.append(k)
	if full_file:
		for k,p in getattr(pipe, "vae").named_parameters():
			t=p.data.to(torch.float32)
			if k in sd:
				sum1=torch.sum(torch.abs(t)).item()
				sum2=torch.sum(torch.abs(sd[k].to(torch.float32))).item()
				n=not(math.isnan(sum1) or math.isnan(sum2))
				if n and sum1!=sum2:
					t=t*sum2/sum1
			osd={}
			if k.startswith("encoder."):
				for key in vae_key2:
					if vae_key2[key] in k:
						k=k.replace(vae_key2[key],key)
			elif k.startswith("decoder."):
				for key in vae_key3:
					if vae_key3[key] in k:
						k=k.replace(vae_key3[key],key)
			else:
				for key in vae_key1:
					if vae_key1[key] in k:
						k=k.replace(vae_key1[key],key)
			k="first_stage_model."+k
			osd[k]=t.to(torch.bfloat16)
			save_file(osd,os.getcwd()+"/safe_temp/"+k+".safetensors")
			keys.append(k)
			
		for k,p in getattr(pipe, "text_encoder").named_parameters():
			t=p.data.to(torch.float32)
			if k in sd:
				sum1=torch.sum(torch.abs(t)).item()
				sum2=torch.sum(torch.abs(sd[k].to(torch.float32))).item()
				n=not(math.isnan(sum1) or math.isnan(sum2))
				if n and sum1!=sum2:
					t=t*sum2/sum1
			osd={}
			k="cond_stage_model.qwen3_06b.transformer.model."+k
			osd[k]=t.to(torch.bfloat16)
			save_file(osd,os.getcwd()+"/safe_temp/"+k+".safetensors")
			keys.append(k)
			
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
	pipe = AnimaPipeline.from_single_file(base_path,torch_dtype=torch.float32)

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
							transformer_sd["lycoris_"+key_dict[k]+"."+k2]=sd.pop(k+"."+k2)
						continue

				m=re.search(r"blocks_[0-9]+_",key_dict[k])
				if m!=None:
					key_dict[k]=key_dict[k][m.start():]
					for k2 in trans_key1:
						mk2=k2.removesuffix(".weight").replace(".","_")
						mk2_value=trans_key1[k2].removesuffix(".weight").replace(".","_")
						key_dict[k]=key_dict[k].replace(mk2_value,mk2)
					for k2 in MODULE_type.weight_list:
						if k+"."+k2 in sd:
							transformer_sd["lycoris_core_transformer_"+key_dict[k]+"."+k2]=sd.pop(k+"."+k2)
					continue

				for k3 in trans_key2:
					mk2=k3.removesuffix(".weight").replace(".","_")
					mk2_value=trans_key2[k3].removesuffix(".weight").replace(".","_")
					m=re.search(mk2,key_dict[k])
					if m!=None:
						key_dict[k]=mk2
						for k2 in MODULE_type.weight_list:
							if k+"."+k2 in sd:
								transformer_sd["lycoris_core_"+key_dict[k]+"."+k2]=sd.pop(k+"."+k2)
						break
					m=re.search(mk2_value,key_dict[k])
					if m!=None:
						key_dict[k]=mk2
						for k2 in MODULE_type.weight_list:
							if k+"."+k2 in sd:
								transformer_sd["lycoris_core_"+key_dict[k]+"."+k2]=sd.pop(k+"."+k2)
						break

			if transformer_sd=={} and text_encoder_sd=={}:
				if win!=None:
					win["RUN"].Update(disabled=False)
					win["info"].update(loras[i]+" isn't supported.")
				else:
					print(loras[i]+" isn't supported.")
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
	except:
		if win!=None:
			win["RUN"].Update(disabled=False)
			win["info"].update("I failed loading lora.")
		else:
			print("I failed loading lora.")
		return

	if win!=None:
		win["info"].update("output ckpt file")
	else:
		print("output ckpt file")
	sd=load_file(base_path)
	sd = _strip_wrapping_prefixes(sd)
	core_state_dict, llm_adapter_state_dict = _convert_anima_state_dict_to_diffusers(sd)
	vsd,tsd=_vae_text_check(base_path)
	sd={**core_state_dict, **llm_adapter_state_dict, **vsd, **tsd}
	keys=zip_ckpt(pipe,sd,full_file)
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
		[sg.InputText(key='ckpt',right_click_menu=grp_rclick_menu["ckpt"]),sg.FileBrowse('select ckpt', file_types=(('ckpt file', '.safetensors'),))],
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
				
	window.close()
