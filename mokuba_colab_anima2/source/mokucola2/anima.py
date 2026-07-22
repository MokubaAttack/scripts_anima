import torch
from diffusers import (
	AnimaModularPipeline,
	FlowMatchLCMScheduler,
	FlowMatchEulerDiscreteScheduler
)
import random
import os
import re
from IPython.display import clear_output
from safetensors.torch import load_file
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

from .meta import plus_meta
from .discord import to_discord
from .imgshow import imgshow
from .imgup import imgup
from .ccc import (
	flush,
	getid,
	safe2diff
)

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

class mokuanipipe:
	def __init__(self):
		self.meta_dict={}
		self.pipe=None

	def mkpipe(
		self,
		base_safe="base.safetensors",
		loras=[],
		lora_weights=[],
		sample="",
		sgm="",
		dtype="f32",
		dev="cuda"
		):
		if not(os.path.exists(base_safe)):
			print("the checkpoint file does not exist.")
			return -1
		if base_safe.endswith(".safetensors"):
			self.meta_dict["ckpt"]=getid(base_safe,None)
			self.meta_dict["ckpt_name"]=base_safe
			base_safe=safe2diff(safe_path=base_safe,id=self.meta_dict["ckpt"])
		else:
			if os.path.exists(base_safe+"/id.txt"):
				f=open(base_safe+"/id.txt","r")
				self.meta_dict["ckpt"]=f.read()
				f.close()
			else:
				self.meta_dict["ckpt"]=""
			self.meta_dict["ckpt_name"]=base_safe
		flush()

		if dtype=="bf16":
			dtype=torch.bfloat16
		elif dtype=="f16":
			dtype=torch.float16
		else:
			dtype=torch.float32

		self.pipe = AnimaModularPipeline.from_pretrained(base_safe)
		self.pipe.load_components(torch_dtype=torch.bfloat16)
		self.pipe.to(dev)
		self.pipe.to(dtype)
		
		if sgm.lower()=="karras":
			sgmuse=[True,False,False]
		elif sgm.lower()=="exponential":
			sgmuse=[False,True,False]
		elif sgm.lower()=="beta":
			sgmuse=[False,False,True]
		else:
			sgm="normal"
			sgmuse=[False,False,False]

		if sample.lower()=="FlowMatch_LCM".lower():
			self.pipe.scheduler=FlowMatchLCMScheduler.from_config(
				self.pipe.scheduler.config,
				use_karras_sigmas=sgmuse[0],
				use_exponential_sigmas=sgmuse[1],
				use_beta_sigmas=sgmuse[2],
			)
		else:
			sample="FlowMatch_Euler"
			self.pipe.scheduler=FlowMatchEulerDiscreteScheduler.from_config(
				self.pipe.scheduler.config,
				use_karras_sigmas=sgmuse[0],
				use_exponential_sigmas=sgmuse[1],
				use_beta_sigmas=sgmuse[2],
			)

		self.meta_dict["sa"]=sample+" "+sgm
		
		if loras!=[]:
			if len(loras)!=len(lora_weights):
				print("the number of lora does not equal the number of lora weight.")
				return -1
			i=0
			meta_id_list=[]
			meta_weight_list=[]
			for line in loras:
				if line.endswith(".safetensors"):
					line=line.replace(".safetensors","")
				if os.path.exists(line+".safetensors"):
					sd=load_file(line+".safetensors")

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
						raise ValueError(line+".safetensors isn't supported.")
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
						raise ValueError(line+".safetensors isn't supported.")
					if transformer_sd!={}:
						wrapper, _ = create_lycoris_from_weights(multiplier=lora_weights[i],file="dummy.safetensors",module=self.pipe.transformer, weights_sd=transformer_sd)
						wrapper.merge_to()
						del wrapper
						flush()
					del transformer_sd
					if text_encoder_sd!={}:
						wrapper, _ = create_lycoris_from_weights(multiplier=lora_weights[i],file="dummy.safetensors",module=self.pipe.text_encoder, weights_sd=text_encoder_sd)
						wrapper.merge_to()
						del wrapper
						flush()
					del text_encoder_sd
					if text_conditioner_sd!={}:
						wrapper, _ = create_lycoris_from_weights(multiplier=lora_weights[i],file="dummy.safetensors",module=self.pipe.text_conditioner, weights_sd=text_conditioner_sd)
						wrapper.merge_to()
						del wrapper
						flush()
					del text_conditioner_sd		
					
					print(line+".safetensors is loaded.")

					list1,list2=getid(line+".safetensors",lora_weights[i])
					meta_id_list=meta_id_list+list1
					meta_weight_list=meta_weight_list+list2
					del list1,list2
				else:
					print(line+".safetensors does not exist.")
					return -1
				i=i+1
			self.meta_dict["lora"]=str(meta_id_list)
			self.meta_dict["w"]=str(meta_weight_list)
			del meta_id_list,meta_weight_list
		else:
			self.meta_dict["lora"]="[]"
		self.meta_dict["loras"]=loras
		self.meta_dict["lora_weights"]=lora_weights
		return 1

	def text2image(self,prompt,n_prompt,gs,step,seed,x,y,out,out_folder,si,url):
		if self.pipe==None:
			print("You must make a pipeline.")
			return []

		self.meta_dict["pr"]=prompt
		self.meta_dict["ne"]=n_prompt
		memo="seed\n"
		for i in seed:
			memo=memo+str(i)+"\n"
		memo=memo+"ckpt : "+self.meta_dict["ckpt_name"]+"\n"
		memo=memo+"scheduler : "+self.meta_dict["sa"]+"\n"
		if self.meta_dict["loras"]!=[]:
			memo=memo+"lora : weight\n"
			for i in range(len(self.meta_dict["loras"])):
				memo=memo+self.meta_dict["loras"][i]+" : "+str(self.meta_dict["lora_weights"][i])+"\n"
		for k in ["gs","hs","ds","hu","hum","up"]:
			if k in self.meta_dict:
				del self.meta_dict[k]
		memo=memo+"num_inference_steps : "+str(step)+"\n"
		self.meta_dict["st"]=str(step)
		memo=memo+"guidance_scale : "+str(gs)+"\n"
		self.meta_dict["cf"]=str(gs)
		memo=memo+"prompt\n"+prompt+"\nnegative prompt\n"+n_prompt+"\n"

		self.pipe.guider.register_to_config(guidance_scale=gs)
		self.pipe.vae.enable_tiling()
		
		if not(os.path.exists(out_folder)) and out:
			os.makedirs(out_folder)
			
		j=0
		images=[]
		for i in seed:
			j=j+1
			clear_output(True)
			print(memo)
			if len(images)>0 and si:
				imgshow(imgs=images)
			image = self.pipe(
				prompt=prompt,
				negative_prompt=n_prompt,
				width=x,
				height=y,
				num_inference_steps=step,
				generator=torch.manual_seed(i),
			).images[0]
			images.append(image)

			if out:
				self.meta_dict["se"]=str(i)
				self.meta_dict["input"]=out_folder+"/"+str(j)+"_"+str(i)+".jpg"
				plus_meta(self.meta_dict,image)
				if url!="":
					to_discord(self.meta_dict["input"],url)
			del image
			flush()

		clear_output(True)
		print(memo)
		if si:
			imgshow(imgs=images)

		return images
		
	def image2imageup(self,prompt,n_prompt,gs,step,seed,x,y,ss,images,out,out_folder,si,url):
		if self.pipe==None:
			print("You must make a pipeline.")
			return []

		self.meta_dict["pr"]=prompt
		self.meta_dict["ne"]=n_prompt
		memo="seed\n"
		for i in seed:
			memo=memo+str(i)+"\n"
		memo=memo+"ckpt : "+self.meta_dict["ckpt_name"]+"\n"
		memo=memo+"scheduler : "+self.meta_dict["sa"]+"\n"
		if self.meta_dict["loras"]!=[]:
			memo=memo+"lora : weight\n"
			for i in range(len(self.meta_dict["loras"])):
				memo=memo+self.meta_dict["loras"][i]+" : "+str(self.meta_dict["lora_weights"][i])+"\n"
		for k in ["gs","hs","ds","hu","hum","up"]:
			if k in self.meta_dict:
				del self.meta_dict[k]
		memo=memo+"guidance_scale : "+str(gs)+"\n"
		self.meta_dict["cf"]=str(gs)
		memo=memo+"prompt\n"+prompt+"\nnegative prompt\n"+n_prompt+"\n"
		
		self.pipe.guider.register_to_config(guidance_scale=gs)
		self.pipe.vae.enable_tiling()
		
		if not(os.path.exists(out_folder)) and out:
			os.makedirs(out_folder)
			
		if self.upscaler==None:
			print("You must make a upscaler.")
			return []
		else:
			checklist=[]
			for j in range(len(images)):
				if images[j]==images[0]:
					checklist.append(1)
			if len(checklist)==len(images):
				u_list=[]
				for j in range(len(images)):
					checklist[j]=False
					u_list.append(1)
				if x!=images[0].width or y!=images[0].height:
					u=x/images[0].width
					images[0]=self.upscaler.run(images[0],x,y)
					for j in range(len(images)):
						u_list[j]=u
						images[j]=images[0]
						checklist[j]=True
			else:
				checklist=[]
				u_list=[]
				for j in range(len(images)):
					if x!=images[j].width or y!=images[j].height:
						u_list.append(x/images[j].width)
						images[j]=self.upscaler.run(images[j],x,y)
						checklist.append(True)
					else:
						u_list.append(1)
						checklist.append(False)

		j=0
		for i in seed:
			j=j+1
			if checklist[j-1]:
				memo1=memo+"Hires steps : "+str(step)+"\n"
				self.meta_dict["hs"]=str(step)
				memo1=memo1+"Denoising strength : "+str(ss)+"\n"
				self.meta_dict["ds"]=str(ss)
				memo1=memo1+"Hires upscale : "+str(u_list[j-1])+"\n"
				self.meta_dict["hu"]=str(u_list[j-1])
				self.meta_dict["hum"],self.meta_dict["up"]=self.upscaler.get_method()
				memo1=memo1+"Hires upscaler : "+self.meta_dict["hum"]+"\n"
			else:
				self.meta_dict["st"]=str(step)
				memo1=memo+"num_inference_steps : "+str(step)+"\n"
				memo1=memo1+"Denoising strength : "+str(ss)+"\n"
				self.meta_dict["ds"]=str(ss)
			clear_output(True)
			print(memo1)
			if len(images)>0 and si:
				imgshow(imgs=images)
				
			image = self.pipe(
				prompt=prompt,
				negative_prompt=n_prompt,
				image=images[j-1],
				num_inference_steps=int(step/ss)+1,
				generator=torch.manual_seed(i),
				strength=ss,
				width=x,
				height=y
			).images[0]
		
			if out:
				self.meta_dict["se"]=str(i)
				self.meta_dict["input"]=out_folder+"/"+str(j)+"_"+str(i)+".jpg"
				plus_meta(self.meta_dict,image)
				if url!="":
					to_discord(self.meta_dict["input"],url)
			images[j-1]=image
			del image
			flush()
			
		clear_output(True)
		print(memo1)
		if si:
			imgshow(imgs=images)

		return images
				
	def mkpipe_upscale(self,path,dev):
		self.upscaler=imgup(path,dev)
