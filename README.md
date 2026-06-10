# scripts_anima
There are scripts that I use when I create images by using anima in diffusers or make models.
These scripts are based on [hdae/diffusers-anima](https://github.com/hdae/diffusers-anima).  
I thank [hdae](https://github.com/hdae) and [claude](https://github.com/claude).
## About the problems of hdae/diffusers-anima
When you make images by using hdae/diffusers-anima module, you have errors to load lora and checkpoint into the pipeline.  
There was a small mistake in their code. Then I modified [hdae/diffusers-anima/src/diffusers_anima/loaders/lora_pipeline.py](https://github.com/hdae/diffusers-anima/blob/main/src/diffusers_anima/loaders/lora_pipeline.py) and [hdae/diffusers-anima/src/diffusers_anima/pipelines/anima/loading.py](https://github.com/hdae/diffusers-anima/blob/main/src/diffusers_anima/pipelines/anima/loading.py).  
When you overwrite their codes in modyfied codes, you can images by using hdae/diffusers-anima module.  
When you don't know it well, please run [modify.py](https://github.com/MokubaAttack/scripts_anima/blob/main/modify.py).
## Supporting LyCORIS
I make the module to support [LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS). In next example, you can use.  
```
wrapper, _ = AnimaPipeline.create_lycoris_from_weights( multiplier = 1.0, file = "lycoris.safetensors", weights_sd = None)
wrapper.merge_to()
```
## [MergeLoraBySVD_anima](https://github.com/MokubaAttack/scripts_anima/tree/main/MergeLoraBySVD_anima)
I make [svd_merge_lora.py](https://github.com/kohya-ss/sd-scripts/blob/main/networks/svd_merge_lora.py) of kohya-ss/sd-scripts to run for anima model.
## [difference_between_ckpts_anima](https://github.com/MokubaAttack/scripts_anima/tree/main/difference_between_ckpts_anima)
I make [extract_lora_from_models.py](https://github.com/kohya-ss/sd-scripts/blob/main/networks/extract_lora_from_models.py) of kohya-ss/sd-scripts to run for anima model.
## [make_safetensors_anima](https://github.com/MokubaAttack/scripts_anima/tree/main/make_safetensors_anima)
It is a script that burns loras in a checkpoint.
## [merge_ckpt_anima](https://github.com/MokubaAttack/scripts_anima/tree/main/merge_ckpt_anima)
It is a script that merge anima checkpoints. This script supports block merge, but I do not understand it well.
## [modify_anima_ckpt](https://github.com/MokubaAttack/scripts_anima/tree/main/modify_anima_ckpt)
It is a script that the ckpt that doesn't be loaded in diffusers_anima makes to be loaded. If you modified diffusers_anima by my suggestion, this is not necessary.
## [mokuba_calob_anima](https://github.com/MokubaAttack/scripts_anima/tree/main/mokuba_colab_anima)
I introduced hdae/diffusers-anima module to mokucola. But you need an abundance of time to make images in Google Colab (you need ~6 minutes to make a 1024-1024 image.), because T4 GPU doesn't support bfloat16.
## Credits
[hdae/diffusers-anima](https://github.com/hdae/diffusers-anima)  
[kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)  
[KohakuBlueleaf/LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS)
