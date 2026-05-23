# difference_between_ckpts_anima
I make [extract_lora_from_models.py](https://github.com/kohya-ss/sd-scripts/blob/main/networks/extract_lora_from_models.py) of kohya-ss/sd-scripts to run for anima model.
python modules
```
pip install git+https://github.com/hdae/diffusers-anima.git
pip install FreeSimpleGUI plyer pyperclip
```
## explanations
difference_between_ckpts_anima.main_part(

- paths=[],  
  list of ckpt file
- dim=4,  
  dim of LoRA
- trans_out=True,  
  whether you output transformer difference
- text_out=True,  
  whether you output text_encoder difference
- out_path=None,  
  filename of output
- win=None  
  window of FreeSimpleGUI
  
)
## Credits
[kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)
