import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.system('huggingface-cli download --resume-download Qwen/Qwen2.5-0.5B-Instruct --local-dir /root/dataDisk/model/qwen/ --local-dir-use-symlinks False')