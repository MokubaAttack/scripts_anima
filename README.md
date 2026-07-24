# scripts_anima
There are scripts that I use when I create images by using anima in diffusers or make models.
These scripts are based on [diffusers](https://github.com/huggingface/diffusers).
## [MergeLoraBySVD_anima](https://github.com/MokubaAttack/scripts_anima/tree/hf_ver/MergeLoraBySVD_anima)
I make [svd_merge_lora.py](https://github.com/kohya-ss/sd-scripts/blob/main/networks/svd_merge_lora.py) of kohya-ss/sd-scripts to run for anima model.
## [difference_between_ckpts_anima](https://github.com/MokubaAttack/scripts_anima/tree/hf_ver/difference_between_ckpts_anima)
I make [extract_lora_from_models.py](https://github.com/kohya-ss/sd-scripts/blob/main/networks/extract_lora_from_models.py) of kohya-ss/sd-scripts to run for anima model.
## [make_safetensors_anima](https://github.com/MokubaAttack/scripts_anima/tree/hf_ver/make_safetensors_anima)
It is a script that burns loras in a checkpoint.
## [merge_ckpt_anima](https://github.com/MokubaAttack/scripts_anima/tree/hf_ver/merge_ckpt_anima)
It is a script that merge anima checkpoints. This script supports block merge, but I do not understand it well.
## [mokuba_calob_anima2](https://github.com/MokubaAttack/scripts_anima/tree/hf_ver/mokuba_colab_anima2)
This is huggingface version of mokuba_colab_anima. But you need an abundance of time to make images in Google Colab (you need ~6 minutes to make a 1024-1024 image.), because T4 GPU doesn't support bfloat16.
## Credits
- [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)  
- [KohakuBlueleaf/LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS)
