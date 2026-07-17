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
from safetensors.torch import load_file
import gc
import torch

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

#unet_keys
#sd - hf
#sdxl
unet_conversion_map1={
	"time_embed.0.": "time_embedding.linear_1.",
	"time_embed.2.": "time_embedding.linear_2.",
	"input_blocks.0.0.": "conv_in.",
	"out.0.": "conv_norm_out.",
	"out.2.": "conv_out.",
	"label_emb.0.0.": "add_embedding.linear_1.",
	"label_emb.0.2.": "add_embedding.linear_2.",
}
unet_conversion_map_resnet={
	"in_layers.0": "norm1",
	"in_layers.2": "conv1",
	"out_layers.0": "norm2",
	"out_layers.3": "conv2",
	"emb_layers.1": "time_emb_proj",
	"skip_connection": "conv_shortcut",
}
unet_conversion_map_layer1=[]
for i in range(3):
	for j in range(2):
		unet_conversion_map_layer1+=[("input_blocks."+str(3 * i + j + 1)+".0.","down_blocks."+str(i)+".resnets."+str(j)+".")]
		if i > 0:
			unet_conversion_map_layer1+=[("input_blocks."+str(3 * i + j + 1)+".1.","down_blocks."+str(i)+".attentions."+str(j)+".")]

	for j in range(3):
		unet_conversion_map_layer1+=[("output_blocks."+str(3 * i + j)+".0.","up_blocks."+str(i)+".resnets."+str(j)+".")]
		if i < 2:
			unet_conversion_map_layer1+=[("output_blocks."+str(3 * i + j)+".1.","up_blocks."+str(i)+".attentions."+str(j)+".")]

	if i < 3:
		unet_conversion_map_layer1+=[("input_blocks."+str(3 * (i + 1))+".0.op.","down_blocks."+str(i)+".downsamplers.0.conv.")]

		if i==0:
			unet_conversion_map_layer1+=[("output_blocks."+str(3 * i + 2)+".1.","up_blocks."+str(i)+".upsamplers.0.")]
		else:
			unet_conversion_map_layer1+=[("output_blocks."+str(3 * i + 2)+".2.","up_blocks."+str(i)+".upsamplers.0.")]
unet_conversion_map_layer1+=[("output_blocks.2.2.conv.","output_blocks.2.1.conv.")]

unet_conversion_map_layer1+=[("middle_block.1.","mid_block.attentions.0.")]
for j in range(2):
	unet_conversion_map_layer1+=[("middle_block."+str(2 * j)+".","mid_block.resnets."+str(j)+".")]
#sd
unet_conversion_map2={
	"time_embed.0.": "time_embedding.linear_1.",
	"time_embed.2.": "time_embedding.linear_2.",
	"input_blocks.0.0.": "conv_in.",
	"out.0.": "conv_norm_out.",
	"out.2.": "conv_out.",
}
unet_conversion_map_layer2=[]
for i in range(4):
	for j in range(2):
		unet_conversion_map_layer2+=[("input_blocks."+str(3 * i + j + 1)+".0.","down_blocks."+str(i)+".resnets."+str(j)+".")]
		if i < 3:
			unet_conversion_map_layer2+=[("input_blocks."+str(3 * i + j + 1)+".1.","down_blocks."+str(i)+".attentions."+str(j)+".")]

	for j in range(3):
		unet_conversion_map_layer2+=[("output_blocks."+str(3 * i + j)+".0.","up_blocks."+str(i)+".resnets."+str(j)+".")]
		if i > 0:
			unet_conversion_map_layer2+=[("output_blocks."+str(3 * i + j)+".1.","up_blocks."+str(i)+".attentions."+str(j)+".")]

	if i < 3:
		unet_conversion_map_layer2+=[("input_blocks."+str(3 * (i + 1))+".0.op.","down_blocks."+str(i)+".downsamplers.0.conv.")]

		if i==0:
			unet_conversion_map_layer2+=[("output_blocks."+str(3 * i + 2)+".1.","up_blocks."+str(i)+".upsamplers.0.")]
		else:
			unet_conversion_map_layer2+=[("output_blocks."+str(3 * i + 2)+".2.","up_blocks."+str(i)+".upsamplers.0.")]

unet_conversion_map_layer2+=[("middle_block.1.","mid_block.attentions.0.")]
for j in range(2):
	unet_conversion_map_layer2+=[("middle_block."+str(2 * j)+".","mid_block.resnets."+str(j)+".")]

def flush():
	gc.collect()
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
	if torch.backends.mps.is_available():
		torch.mps.empty_cache()
	if torch.xpu.is_available():
		torch.xpu.empty_cache()

def load_weight_anima(pipe_module,lora_path,lora_weight):
	sd=load_file(lora_path)

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
		raise ValueError(lora_path+" isn't supported.")
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
			for k2 in block_maps:
				mk2=k2.removesuffix(".weight").replace(".","_")
				mk2_value=block_maps[k2].removesuffix(".weight").replace(".","_")
				key_dict[k]=key_dict[k].replace(mk2,mk2_value)
			for k2 in MODULE_type.weight_list:
				if k+"."+k2 in sd:
					transformer_sd["lycoris_core_transformer_"+key_dict[k]+"."+k2]=sd.pop(k+"."+k2)
			continue

		for k3 in root_map:
			mk2=k3.removesuffix(".weight").replace(".","_")
			mk2_value=root_map[k3].removesuffix(".weight").replace(".","_")
			m=re.search(mk2,key_dict[k])
			if m!=None:
				key_dict[k]=mk2_value
				for k2 in MODULE_type.weight_list:
					if k+"."+k2 in sd:
						transformer_sd["lycoris_core_"+key_dict[k]+"."+k2]=sd.pop(k+"."+k2)
				break
			m=re.search(mk2_value,key_dict[k])
			if m!=None:
				key_dict[k]=mk2_value
				for k2 in MODULE_type.weight_list:
					if k+"."+k2 in sd:
						transformer_sd["lycoris_core_"+key_dict[k]+"."+k2]=sd.pop(k+"."+k2)
				break

	if transformer_sd=={} and text_encoder_sd=={}:
		raise ValueError(lora_path+" isn't supported.")
	if transformer_sd!={}:
		wrapper, _ = create_lycoris_from_weights(multiplier=lora_weight,file="dummy.safetensors",module=pipe_module.transformer, weights_sd=transformer_sd)
		wrapper.merge_to()
		del wrapper
	del transformer_sd
	flush()
	if text_encoder_sd!={}:
		wrapper, _ = create_lycoris_from_weights(multiplier=lora_weight,file="dummy.safetensors",module=pipe_module.text_encoder, weights_sd=text_encoder_sd)
		wrapper.merge_to()
		del wrapper
	del text_encoder_sd
	flush()

def load_weight_sdxl(pipe_module,lora_path,lora_weight):
	sd=load_file(lora_path)

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
		raise ValueError(lora_path+" isn't supported.")
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

	usd={}
	t1sd={}
	t2sd={}

	for k in unet_conversion_map1:
		m="lora_unet_"+k.removesuffix(".").replace(".","_")
		if m in key_dict.values():
			for k3 in key_dict:
				if key_dict[k3]==m:
					k4=k3
					break
			del key_dict[k4]
			for k2 in MODULE_type.weight_list:
				m2=k4+"."+k2
				if m2 in sd:
					usd["lycoris_"+unet_conversion_map1[k].removesuffix(".").replace(".","_")+"."+k2]=sd.pop(m2)

	for k in key_dict:
		m=k.replace(".","_")
		if k.startswith("lora_unet_"):
			m=m.removeprefix("lora_unet_")
			m=m.replace("output_blocks_2_2_conv","up_blocks_0_upsamplers_0_conv")
			for k2 in unet_conversion_map_layer1:
				k3=k2[0].removesuffix(".").replace(".","_")
				m=m.replace(k3,k2[1].removesuffix(".").replace(".","_"))
			if "resnets" in m:
				for k2 in unet_conversion_map_resnet:
					m=m.replace(k2.replace(".","_"),unet_conversion_map_resnet[k2])
			for k2 in MODULE_type.weight_list:
				if k+"."+k2 in sd:
					usd["lycoris_"+m+"."+k2]=sd[k+"."+k2]
		elif k.startswith("lora_te1_"):
			m=m.removeprefix("lora_te1_")
			for k2 in MODULE_type.weight_list:
				if k+"."+k2 in sd:
					t1sd["lycoris_"+m+"."+k2]=sd[k+"."+k2]
		elif k.startswith("lora_te2_"):
			m=m.removeprefix("lora_te2_")
			for k2 in MODULE_type.weight_list:
				if k+"."+k2 in sd:
					t2sd["lycoris_"+m+"."+k2]=sd[k+"."+k2]

	if usd=={} and t1sd=={} and t2sd=={}:
		raise ValueError(lora_path+" isn't supported.")

	if usd!={}:
		wrapper, _ = create_lycoris_from_weights(multiplier=lora_weight,file="dummy.safetensors",module=pipe_module.unet, weights_sd=usd)
		wrapper.merge_to()
	del usd
	flush()

	if t1sd!={}:
		wrapper, _ = create_lycoris_from_weights(multiplier=lora_weight,file="dummy.safetensors",module=pipe_module.text_encoder, weights_sd=t1sd)
		wrapper.merge_to()
	del t1sd
	flush()

	if t2sd!={}:
		wrapper, _ = create_lycoris_from_weights(multiplier=lora_weight,file="dummy.safetensors",module=pipe_module.text_encoder_2, weights_sd=t2sd)
		wrapper.merge_to()
	del t2sd
	flush()

def load_weight_sd(pipe_module,lora_path,lora_weight):
	sd=load_file(lora_path)

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
		raise ValueError(lora_path+" isn't supported.")
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

	usd={}
	t1sd={}

	for k in unet_conversion_map2:
		m="lora_unet_"+k.removesuffix(".").replace(".","_")
		if m in key_dict.values():
			for k3 in key_dict:
				if key_dict[k3]==m:
					k4=k3
					break
			del key_dict[k4]
			for k2 in MODULE_type.weight_list:
				m2=k4+"."+k2
				if m2 in sd:
					usd["lycoris_"+unet_conversion_map2[k].removesuffix(".").replace(".","_")+"."+k2]=sd.pop(m2)

	for k in key_dict:
		m=k.replace(".","_")
		if k.startswith("lora_unet_"):
			m=m.removeprefix("lora_unet_")
			m=m.replace("output_blocks_2_2_conv","up_blocks_0_upsamplers_0_conv")
			for k2 in unet_conversion_map_layer2:
				k3=k2[0].removesuffix(".").replace(".","_")
				m=m.replace(k3,k2[1].removesuffix(".").replace(".","_"))
			if "resnets" in m:
				for k2 in unet_conversion_map_resnet:
					m=m.replace(k2.replace(".","_"),unet_conversion_map_resnet[k2])
			for k2 in MODULE_type.weight_list:
				if k+"."+k2 in sd:
					usd["lycoris_"+m+"."+k2]=sd[k+"."+k2]
		elif k.startswith("lora_te_"):
			m=m.removeprefix("lora_te_")
			for k2 in MODULE_type.weight_list:
				if k+"."+k2 in sd:
					t1sd["lycoris_"+m+"."+k2]=sd[k+"."+k2]

	if usd=={} and t1sd=={}:
		raise ValueError(lora_path+" isn't supported.")

	if usd!={}:
		wrapper, _ = create_lycoris_from_weights(multiplier=lora_weight,file="dummy.safetensors",module=pipe_module.unet, weights_sd=usd)
		wrapper.merge_to()
	del usd
	flush()

	if t1sd!={}:
		wrapper, _ = create_lycoris_from_weights(multiplier=lora_weight,file="dummy.safetensors",module=pipe_module.text_encoder, weights_sd=t1sd)
		wrapper.merge_to()
	del t1sd
	flush()
