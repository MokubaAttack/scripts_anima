import os
os.environ["HF_HOME"]=os.getcwd()+"/pipecache"
from diffusers_anima import AnimaPipeline
from safetensors.torch import (
	load_file,
	save_file
)
import shutil
import torch

CLAMP_QUANTILE=0.99

_ANIMA_BLOCK_MODULE_MAP = {
	"self_attn.q_proj": "attn1.to_q",
	"self_attn.k_proj": "attn1.to_k",
	"self_attn.v_proj": "attn1.to_v",
	"self_attn.output_proj": "attn1.to_out.0",
	"cross_attn.q_proj": "attn2.to_q",
	"cross_attn.k_proj": "attn2.to_k",
	"cross_attn.v_proj": "attn2.to_v",
	"cross_attn.output_proj": "attn2.to_out.0",
	"mlp.layer1": "ff.net.0.proj",
	"mlp.layer2": "ff.net.2",
	"adaln_modulation_self_attn.1": "norm1.linear_1",
	"adaln_modulation_self_attn.2": "norm1.linear_2",
	"adaln_modulation_cross_attn.1": "norm2.linear_1",
	"adaln_modulation_cross_attn.2": "norm2.linear_2",
	"adaln_modulation_mlp.1": "norm3.linear_1",
	"adaln_modulation_mlp.2": "norm3.linear_2",
}

_ANIMA_ROOT_MODULE_MAP = {
	"x_embedder.proj.1": "patch_embed.proj",
	"t_embedder.1.linear_1": "time_embed.t_embedder.linear_1",
	"t_embedder.1.linear_2": "time_embed.t_embedder.linear_2",
	"final_layer.adaln_modulation.1": "norm_out.linear_1",
	"final_layer.adaln_modulation.2": "norm_out.linear_2",
	"final_layer.linear": "proj_out",
}

def split_pipe(path,i,trans_out,text_out):
	pipe=AnimaPipeline.from_single_file(path)
	keys=[]
	if trans_out:
		for name, module in pipe.transformer.named_modules():
			if module.__class__.__name__ in ["CosmosTransformer3DModel"]:
				for child_name, child_module in module.named_modules():
					is_linear = child_module.__class__.__name__ == "Linear"
					is_conv2d = child_module.__class__.__name__ == "Conv2d"
					is_conv2d_1x1 = is_conv2d and child_module.kernel_size == (1, 1)
					if is_linear or is_conv2d:
						key=name + "." + child_name
						if key.startswith("core.transformer_"):
							for k2 in _ANIMA_BLOCK_MODULE_MAP:
								if _ANIMA_BLOCK_MODULE_MAP[k2] in key:
									key=key.replace(_ANIMA_BLOCK_MODULE_MAP[k2],k2)
								key=key.replace("core.transformer_","diffusion_model.")
						else:
							for k2 in _ANIMA_ROOT_MODULE_MAP:
								if _ANIMA_ROOT_MODULE_MAP[k2] in key:
									key=key.replace(_ANIMA_ROOT_MODULE_MAP[k2],k2)
								key=key.replace("core.","diffusion_model.")
						loras={}
						loras[key]=child_module.weight.contiguous()
						save_file(loras,os.getcwd()+"/safe_temp/"+str(i)+"_"+key+".safetensors")
						keys.append(key)
			elif module.__class__.__name__ in ["_LLMAdapter"]:
				for child_name, child_module in module.named_modules():
					is_linear = child_module.__class__.__name__ == "Linear"
					is_conv2d = child_module.__class__.__name__ == "Conv2d"
					is_conv2d_1x1 = is_conv2d and child_module.kernel_size == (1, 1)
					if is_linear or is_conv2d:
						key="diffusion_model."+name + "." + child_name
						loras={}
						loras[key]=child_module.weight.contiguous()
						save_file(loras,os.getcwd()+"/safe_temp/"+str(i)+"_"+key+".safetensors")
						keys.append(key)
	
	if text_out:
		for name, module in pipe.text_encoder.named_modules():
			if module.__class__.__name__ in ["Qwen3Attention","Qwen3MLP"]:
				for child_name, child_module in module.named_modules():
					is_linear = child_module.__class__.__name__ == "Linear"
					is_conv2d = child_module.__class__.__name__ == "Conv2d"
					is_conv2d_1x1 = is_conv2d and child_module.kernel_size == (1, 1)
					if is_linear or is_conv2d:
						key="text_encoders.qwen3_06b.transformer.model."+name + "." + child_name
						loras={}
						loras[key]=child_module.weight.contiguous()
						save_file(loras,os.getcwd()+"/safe_temp/"+str(i)+"_"+key+".safetensors")
						keys.append(key)
	
	return keys

def main_part(
	paths=[],
	dim=4,
	trans_out=True,
	text_out=True,
	out_path=None,
	win=None
):
	if win!=None:
		win["RUN"].Update(disabled=True)

	if not(out_path.endswith(".safetensors")):
		if win==None:
			print("the output path is needed to be a safetensors file.")
		else:
			win["RUN"].Update(disabled=False)
			win["info"].update("the output path is needed to be a safetensors file.")
		return

	for path in paths:
		if not(os.path.exists(path)):
			if win==None:
				print(path+" does not exist.")
			else:
				win["RUN"].Update(disabled=False)
				win["info"].update(os.path.basename(path)+" does not exist.")
			return
			
	if trans_out==False and text_out==False:
		if win==None:
			print("You choose no contents.")
		else:
			win["RUN"].Update(disabled=False)
			win["info"].update("You choose no contents.")
		return
		
	try:
		temp_path=os.getcwd()+"/safe_temp"
		if os.path.exists(temp_path):
			shutil.rmtree(temp_path)
		os.mkdir(temp_path)
		
		if win==None:
			print("load "+paths[0])
		else:
			win["info"].update("load "+paths[0])
		keys1=split_pipe(paths[0],0,trans_out,text_out)
		
		if win==None:
			print("load "+paths[1])
		else:
			win["info"].update("load "+paths[1])
		keys2=split_pipe(paths[1],1,trans_out,text_out)
		
		names=list(set(keys1+keys2))
		sd={}
		names_sum=len(names)
		name_count=0

		for name in names:
			name_count=name_count+1
			if win!=None:
				win["info"].update("differ "+str(name_count)+"/"+str(names_sum))
			else:
				print("\rdiffer "+str(name_count)+"/"+str(names_sum),end="")
			sd1=load_file(temp_path+"/0_"+name+".safetensors")
			sd2=load_file(temp_path+"/1_"+name+".safetensors")
			mat=(sd2[name]-sd1[name]).to(torch.float)
			
			if torch.any(torch.isnan(mat)):
				continue

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

			try:
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
			except:
				if win==None:
					print("")
					print(name)
				else:
					win["RUN"].Update(disabled=False)
					win["info"].update("error "+name)
				shutil.rmtree(temp_path)
				return

			up_weight = U
			down_weight = Vh

			sd[name + ".lora_B.weight"] = up_weight.to(torch.float16).contiguous()
			sd[name + ".lora_A.weight"] = down_weight.to(torch.float16).contiguous()
			del sd1,sd2,mat,up_weight,down_weight,U,S,Vh,dist,hi_val,low_val
			gc.collect()
		save_file(sd,out_path)
		shutil.rmtree(temp_path)
		if win==None:
			print("")
			print("fin")
		else:
			win["RUN"].Update(disabled=False)
			win["info"].update("fin")
	except:
		if os.path.exists(temp_path):
			shutil.rmtree(temp_path)
		if win==None:
			print("I failed in the output.")
		else:
			win["RUN"].Update(disabled=False)
			win["info"].update("I failed in the output.")
			
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
		[sg.Text("model_org"), sg.Input(key="ckpt1",right_click_menu=grp_rclick_menu["ckpt1"]),sg.FileBrowse( file_types=(('ckpt file', '.safetensors'),))],
		[sg.Text("model_tuned"), sg.Input(key="ckpt2",right_click_menu=grp_rclick_menu["ckpt2"]),sg.FileBrowse( file_types=(('ckpt file', '.safetensors'),))],
		[sg.Text("dim"), sg.Input(key="dim",right_click_menu=grp_rclick_menu["dim"])],
		[sg.Text("output path"), sg.Input(key="out",right_click_menu=grp_rclick_menu["out"]),sg.FileSaveAs(file_types=(('ckpt file', '.safetensors'),))],
		[sg.Checkbox('transformer', default=True, key='unet'),sg.Checkbox('text_encoder', default=True, key='text1')],
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
				
				thread1 = threading.Thread(target=diff_ckpt,args=(paths,dim,trans_out,text_out,out_path,window))
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
