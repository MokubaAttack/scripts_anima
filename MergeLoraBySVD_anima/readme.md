# MergeLoraBySVD_anima
I make [svd_merge_lora.py](https://github.com/kohya-ss/sd-scripts/blob/main/networks/svd_merge_lora.py) of kohya-ss/sd-scripts to run for anima model.
## requirements
python modules
```
pip install git+https://github.com/hdae/diffusers-anima.git
pip install FreeSimpleGUI pyperclip
```
## explanations
MergeLoraBySVD.main_part(

-	loras=[],  
  list of lora file
-	weights=[],  
  list of lora weight
-	precision="float",  
  precision of calculation ( "float", "fp16", "bf16" )
-	save_precision="fp16",  
  precision of output file ( "float", "fp16", "bf16" )
-	new_rank=16,  
  dim of LoRA
-	new_conv_rank=None,  
  dim of Conv2d 3x3 LoRA ( When it is None, it is same to new_rank )
-	device=None,  
  When you input "cuda", it calculates by GPU.
-	save_to=None,  
  filename of output
-	win=None,  
  window of FreeSimpleGUI
-	meta_dict=None,  
  the dict of metadata
-	dof=False  
  whether you delete original files
 
)
## Credits
[kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)
