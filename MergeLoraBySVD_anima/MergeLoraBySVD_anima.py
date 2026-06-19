from safetensors.torch import (
	load_file,
	save_file
)
import torch
import os

placeholders = {
	"adaln_modulation_self_attn":"moku01",
	"adaln_modulation_cross_attn":"moku02",
	"adaln_modulation_mlp":"moku03",
	"self_attn":"moku04",
	"cross_attn":"moku05",
	"output_proj":"moku06",
	"q_proj":"moku07",
	"k_proj":"moku08",
	"v_proj":"moku09",
	"x_embedder.proj.1": "moku10",
	"t_embedder.1.linear_1": "moku11",
	"t_embedder.1.linear_2": "moku12",
	"final_layer.adaln_modulation.1": "moku13",
	"final_layer.adaln_modulation.2": "moku14",
	"final_layer.linear": "moku15",
}
placeholders2 = {
	"embed_tokens":"moku01",
	"input_layernorm":"moku02",
	"down_proj":"moku03",
	"gate_proj":"moku04",
	"up_proj":"moku05",
	"post_attention_layernorm":"moku06",
}
endkeys=(".lora_A.weight",".lora_B.weight",".lora_down.weight",".lora_up.weight",".alpha")

CLAMP_QUANTILE=0.99

def convkey(raw_path):
	sd=load_file(raw_path)
	sd_keys=list(sd)
	c=0
	for path in sd_keys:
		if not(path.endswith((".lora_A.weight",".lora_down.weight"))):
			if not(path.endswith(endkeys)):
				sd.pop(path)
			continue
		c=c+1
		if ".lora_A.weight" in path:
			path=path.removesuffix(".lora_A.weight")
		else:
			path=path.removesuffix(".lora_down.weight")

		if path.startswith("diffusion_model.") or path.startswith("text_encoders.qwen3_06b.transformer.model."):
			mpath=path
		elif path.startswith("lora_te_"):
			mpath = path.removeprefix("lora_te_")
			for key, token in placeholders2.items():
				mpath =mpath.replace(key, token)
			mpath = mpath.replace("_", ".")
			for key, token in placeholders2.items():
				mpath = mpath.replace(token, key)
			mpath= "text_encoders.qwen3_06b.transformer.model."+mpath
		elif path.startswith("lora_unet_"):
			mpath = path.removeprefix("lora_unet_")
			for key, token in placeholders.items():
				mpath =mpath.replace(key, token)
			mpath = mpath.replace("_", ".")
			for key, token in placeholders.items():
				mpath = mpath.replace(token, key)
			mpath= "diffusion_model."+path
		else:
			return None

		for k in endkeys:
			if path+k in sd:
				w=sd.pop(path+k)
				if k==".lora_down.weight":
					k=".lora_A.weight"
				elif k==".lora_up.weight":
					k=".lora_B.weight"
				sd[mpath+k]=w

	if c==0:
		return None
	return sd

def str_to_dtype(p):
	if p == "float":
		return torch.float
	if p == "fp16":
		return torch.float16
	if p == "bf16":
		return torch.bfloat16
	return None

def main_part(
	loras=[],
	weights=[],
	precision="float",
	save_precision="fp16",
	new_rank=16,
	new_conv_rank=None,
	device=None,
	save_to=None,
	win=None,
	meta_dict=None,
	dof=False
):
	if win!=None:
		win['RUN'].Update(disabled=True)
	if len(loras) != len(weights):
		if win==None:
			print("number of models must be equal to number of ratios.")
		else:
			win["info"].update("number of models must be equal to number of ratios.")
			win['RUN'].Update(disabled=False)
		return

	try:
		merge_dtype = str_to_dtype(precision)
		save_dtype = str_to_dtype(save_precision)
		if save_dtype is None:
			save_dtype = merge_dtype

		new_conv_rank = new_conv_rank if new_conv_rank is not None else new_rank
		sds=[]
		keys=[]
		for lora in loras:
			sd=convkey(lora)
			if sd==None:
				if win==None:
					print(os.path.basename(lora)+" isn't supported.")
				else:
					win["info"].update(os.path.basename(lora)+" isn't supported.")
					win['RUN'].Update(disabled=False)
				return
			sds.append(sd)
			keys=keys+list(sd)

		keys=list(set(keys))
		key_sum=len(keys)
		key_count=0
		merged_lora_sd={}

		if win==None:
			print("svd")

		for k in keys:
			key_count=key_count+1
			if win!=None:
				win["info"].update("svd : "+str(key_count)+"/"+str(key_sum))
			else:
				print("\r"+str(key_count)+"/"+str(key_sum),end="")
			if not(k.endswith(".lora_A.weight")):
				continue

			mat=0
			for i in range(len(sds)):
				if not(k in sds[i]):
					continue
				wa=sds[i].pop(k)
				wb=sds[i].pop(k.replace(".lora_A.weight",".lora_B.weight"))

				network_dim = wa.size()[0]
				if k.replace(".lora_A.weight",".alpha") in sds[i]:
					alpha=sds[i].pop(k.replace(".lora_A.weight",".alpha"))
				else:
					alpha=network_dim
				in_dim = wa.size()[1]
				out_dim = wb.size()[0]
				conv2d = len(wa.size()) == 4
				kernel_size = None if not conv2d else wa.size()[2:4]
				scale = alpha / network_dim

				if type(mat) is int:
					mat = torch.zeros((out_dim, in_dim, *kernel_size) if conv2d else (out_dim, in_dim), dtype=merge_dtype)

				if device:
					mat = mat.to(device)
					wb = wb.to(device)
					wa = wa.to(device)
					scale = scale.to(device)
					
				if not conv2d:
					mat = mat + weights[i] * (wb @ wa) * scale
				elif kernel_size == (1, 1):
					mat = (
						mat
						+ weights[i]
						* (wb.squeeze(3).squeeze(2) @ wa.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
						* scale
					)
				else:
					conved = torch.nn.functional.conv2d(wa.permute(1, 0, 2, 3), wb).permute(1, 0, 2, 3)
					mat = mat + weights[i] * conved * scale

			conv2d = len(mat.size()) == 4
			kernel_size = None if not conv2d else mat.size()[2:4]
			conv2d_3x3 = conv2d and kernel_size != (1, 1)
			out_dim, in_dim = mat.size()[0:2]

			if conv2d:
				if conv2d_3x3:
					mat = mat.flatten(start_dim=1)
				else:
					mat = mat.squeeze()

			module_new_rank = new_conv_rank if conv2d_3x3 else new_rank
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

			merged_lora_sd[k.replace(".lora_A.weight",".lora_B.weight")] = up_weight.to("cpu").contiguous()
			merged_lora_sd[k] = down_weight.to("cpu").contiguous()

		for key in list(merged_lora_sd):
			value = merged_lora_sd[key]
			if type(value) == torch.Tensor and value.dtype.is_floating_point and value.dtype != save_dtype:
				merged_lora_sd[key] = value.to(save_dtype)

		if isinstance(meta_dict, dict):
			save_file(merged_lora_sd,save_to,metadata=meta_dict)
		else:
			save_file(merged_lora_sd,save_to)
		if dof:
			for lora in loras:
				os.remove(lora)
		if win==None:
			print("")
			print("fin")
		else:
			win["info"].update("fin")
			win['RUN'].Update(disabled=False)
	except:
		if win==None:
			print("error")
		else:
			win["info"].update("error")
			win['RUN'].Update(disabled=False)

if __name__=="__main__":
	import threading
	import tkinter as tk
	import pyperclip
	import FreeSimpleGUI as sg

	sg.theme('GrayGrayGray')
	grp_rclick_menu={}
	keys=["ckpt1","ckpt2","ckpt3","ckpt4","w1","w2","w3","w4","id1","id2","id3","id4","out","d"]
	for key in keys:
		grp_rclick_menu[key]=[
			"",
			[
				"-copy-::"+key,"-cut-::"+key,"-paste-::"+key
			]
		]		
			
	layout=[
		[sg.Text("lora1"), sg.Input(key="ckpt1",right_click_menu=grp_rclick_menu["ckpt1"]),sg.FileBrowse( file_types=(('lora file', '.safetensors'),)),sg.Button('clear', key='clear1')],
		[sg.Text("weight"),sg.Input("0.7",key="w1",right_click_menu=grp_rclick_menu["w1"], size=(10, 1)),sg.Text("id"),sg.Input("",key="id1",right_click_menu=grp_rclick_menu["id1"], size=(20, 1))],
		[sg.Text("lora2"), sg.Input(key="ckpt2",right_click_menu=grp_rclick_menu["ckpt2"]),sg.FileBrowse( file_types=(('lora file', '.safetensors'),)),sg.Button('clear', key='clear2')],
		[sg.Text("weight"),sg.Input("0.7",key="w2",right_click_menu=grp_rclick_menu["w2"], size=(10, 1)),sg.Text("id"),sg.Input("",key="id2",right_click_menu=grp_rclick_menu["id2"], size=(20, 1))],
		[sg.Text("lora3"), sg.Input(key="ckpt3",right_click_menu=grp_rclick_menu["ckpt3"]),sg.FileBrowse( file_types=(('lora file', '.safetensors'),)),sg.Button('clear', key='clear3')],
		[sg.Text("weight"),sg.Input("0.7",key="w3",right_click_menu=grp_rclick_menu["w3"], size=(10, 1)),sg.Text("id"),sg.Input("",key="id3",right_click_menu=grp_rclick_menu["id3"], size=(20, 1))],
		[sg.Text("lora4"), sg.Input(key="ckpt4",right_click_menu=grp_rclick_menu["ckpt4"]),sg.FileBrowse( file_types=(('lora file', '.safetensors'),)),sg.Button('clear', key='clear4')],
		[sg.Text("weight"),sg.Input("0.7",key="w4",right_click_menu=grp_rclick_menu["w4"], size=(10, 1)),sg.Text("id"),sg.Input("",key="id4",right_click_menu=grp_rclick_menu["id4"], size=(20, 1))],
		[sg.Text("dim"), sg.Input("16",key="d",right_click_menu=grp_rclick_menu["d"])],
		[sg.Text("output path"), sg.Input(key="out",right_click_menu=grp_rclick_menu["out"]),sg.FileSaveAs(file_types=(('lora file', '.safetensors'),)),sg.Button('clear', key='clear_out')],
		[sg.Checkbox('del original files', key='dof')],
		[sg.Text("infomation",key="info")],
		[sg.Button('RUN', key='RUN'),sg.Button('EXIT', key='EXIT')]
	]

	window = sg.Window('merge lora anima', layout,keep_on_top=True)

	while True:
		event, values = window.read()
		if event == sg.WINDOW_CLOSED:
			break
		elif event=="EXIT":
			break
		elif event=="RUN":
			c1=not(values["ckpt1"]=="" and values["ckpt2"]=="" and values["ckpt3"]=="" and values["ckpt4"]=="")
			if c1:
				if values["out"]=="":
					names=[]
					for i in range(4):
						if values["ckpt"+str(i+1)]!="":
							names.append(values["ckpt"+str(i+1)])
					out_path=""
					for line in names:
						if out_path=="":
							out_path=os.path.dirname(line)+"/"+os.path.basename(line).split(".")[0]
						else:
							out_path=out_path+"_"+os.path.basename(line).split(".")[0]
					out_path=out_path+".safetensors"
					window["out"].update(out_path)
				else:
					out_path=values["out"]
					if not(out_path.endswith(".safetensors")):
						out_path=out_path+".safetensors"
						window["out"].update(out_path)

				loras=[]
				weights=[]
				ids=[]
				for i in range(4):
					if values["ckpt"+str(i+1)]!="":
						loras.append(values["ckpt"+str(i+1)])
						try:
							weights.append(float(values["w"+str(i+1)]))
						except:
							weights.append(0.7)
						window["w"+str(i+1)].update(str(weights[-1]))
						try:
							ids.append(int(values["id"+str(i+1)]))
						except:
							ids.append(5)
						window["id"+str(i+1)].update(str(ids[-1]))
				
				try:
					dim=int(values["d"])
				except:
					dim=16
				window["d"].update(str(dim))

				meta={}
				meta["id"]=str(ids).replace("[","").replace("]","").replace(" ","")
				meta["weight"]=str(weights).replace("[","").replace("]","").replace(" ","")

				ok = sg.popup_ok_cancel(out_path,title='output file',keep_on_top=True)
				if ok=="OK":
					thread1 = threading.Thread(target=main_part,args=(loras,weights,"float","bf16",dim,None,None,out_path,window,meta,values["dof"]))
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
		elif "clear" in event:
			try:
				if event=="clear_out":
					key="out"
				else:
					key="ckpt"+event.replace("clear","")
					key2="id"+event.replace("clear","")
					window[key2].update("")
				window[key].update("")
			except:
				pass

	window.close()
