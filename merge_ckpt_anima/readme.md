# merge_ckpt_anima
It is a script that merge anima checkpoints. This script supports block merge, but I do not understand it well.
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
2. Select the checkpoint files ( .safetensors file ).
3. Input weights of second checkpoint file. If you input 0.3, the weight of first model is 0.7 and the weight of second model is 0.3.
4. Input the output path ( .safetensors file ).
5. Click run button.
6. After a while, the output file is generated.
