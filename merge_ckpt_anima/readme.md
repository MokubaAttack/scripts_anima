# merge_ckpt_anima
It is a script that merge anima checkpoints. This script supports block merge, but I do not understand it well.
## requirements
python modules
```
pip install git+https://github.com/hdae/diffusers-anima.git
pip install FreeSimpleGUI pyperclip
```
## How to use
1. Run this script.
2. Select the checkpoint files ( .safetensors file ).
3. Input weights of second checkpoint file. If you input 0.3, the weight of first model is 0.7 and the weight of second model is 0.3.
4. Input the output path ( .safetensors file ).
5. Click run button.
6. After a while, the output file is generated.
