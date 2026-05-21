import subprocess

cmd=["pip","install","git+https://github.com/hdae/diffusers-anima.git"]
subprocess.run(cmd)

import torch
import os
import requests

anima_path=torch.__file__.replace("\\","/")
anima_path=anima_path.replace("torch/__init__.py","diffusers_anima")

path1=anima_path+"/loaders/lora_pipeline.py"
url1="https://raw.githubusercontent.com/MokubaAttack/scripts_anima/refs/heads/main/lora_pipeline.py"
response = requests.get(url1)
with open(path1, 'wb') as f:
    f.write(response.content)

path2=anima_path+"/pipelines/anima/loading.py"
url2="https://raw.githubusercontent.com/MokubaAttack/scripts_anima/refs/heads/main/loading.py"
response = requests.get(url2)
with open(path2, 'wb') as f:
    f.write(response.content)
