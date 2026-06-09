import torch
from diffusers_anima import AnimaPipeline
import random
import os
from IPython.display import clear_output
from safetensors.torch import load_file

from .meta import plus_meta
from .discord import to_discord
from .imgshow import imgshow
from .imgup import imgup
from .ccc import (
	flush,
	getid
)

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
		self.meta_dict["ckpt"]=getid(base_safe,None)
		self.meta_dict["ckpt_name"]=base_safe

		if dtype=="bf16":
			dtype=torch.bfloat16
		elif dtype=="f16":
			dtype=torch.float16
		else:
			dtype=torch.float32

		self.pipe = AnimaPipeline.from_single_file(base_safe,torch_dtype=dtype)
		self.pipe.to(dev)
		
		if sample in ["euler","euler_a_rf","euler_ancestral_rf"]:
			if not(sgm in ["beta","simple","normal"]):
				sgm="normal"
		else:
			sample=="flowmatch_euler"
			sgm="uniform"
		self.meta_dict["sa"]=sample+" "+sgm
		self.pipe.scheduler.set_sampling_config(sampler=sample,sigma_schedule=sgm)
		
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
					lora_check=True
					for k in sd:
						if k.endswith(".lokr_w1") or k.endswith(".lokr_w1_a"):
							lora_check=False
							break
							
					if lora_check:
						self.pipe.load_lora_weights(line+".safetensors",adapter_name="style"+str(i))
						self.pipe.set_adapters("style"+str(i), adapter_weights=[lora_weights[i]])
						self.pipe.fuse_lora()
						self.pipe.unload_lora_weights()
					else:
						wrapper,_=self.pipe.create_lycoris_from_weights(multiplier=lora_weights[i],weights_sd=sd)
						wrapper.merge_to()
					
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
				guidance_scale=gs,
				generator=torch.manual_seed(i),
			).images[0]
			images.append(image)

			if out:
				self.meta_dict["se"]=str(i)
				self.meta_dict["input"]=out_folder+"/"+str(j)+"_"+str(i)+".png"
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
				guidance_scale=gs,
				num_inference_steps=int(step/ss)+1,
				generator=torch.manual_seed(i),
				strength=ss,
				width=x,
				height=y
			).images[0]
		
			if out:
				self.meta_dict["se"]=str(i)
				self.meta_dict["input"]=out_folder+"/"+str(j)+"_"+str(i)+".png"
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
