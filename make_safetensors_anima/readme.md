# make_safetensors_anima
It is a script that burns loras in a checkpoint.
## requirements
python modules
```
diffusers==0.39.0
transformers==5.11.0
torch==2.11.0
torchvision==0.26.0
accelerate
lycoris-lora
FreeSimpleGUI
pyperclip
compel
realesrgan
optimum-quanto
piexif
```
## How to use
1. Run this script.
2. Select the checkpoint file ( .safetensors file ).
3. Input loras that you want to burn in the checkpoint.  
4. Input the output path ( .safetensors file ).
5. Click run button.
6. After a while, the output file is generated.
