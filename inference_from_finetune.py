import json
import torch
import colossalai
from tqdm import tqdm
from colossalai.booster import Booster
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer.hybrid_adam import HybridAdam
from transformers import AutoModelForCausalLM, AutoTokenizer
from colossalai.booster.plugin import HybridParallelPlugin


# load model
model_name = "/root/dataDisk/model/qwen"
data_path = "/root/commonData/Wukong/wukong.jsonl"
model_ckpt_path = "/root/dataDisk/model/qwen_save"
optimizer_ckpt_path = "/root/dataDisk/model/qwen_optim_save"

colossalai.launch_from_torch()
coordinator = DistCoordinator()

# init plugin & booster
plugin = HybridParallelPlugin(
    tp_size=1,
    pp_size=1,
    zero_stage=1,
    enable_fused_normalization=torch.cuda.is_available(),
    microbatch_size=1,
    precision="bf16",
)

booster = Booster(plugin=plugin)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="cuda"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
optimizer = HybridAdam(model.parameters())

# boost your model 
model, optimizer, _, _, _ = booster.boost(model, optimizer)


# load model & optimizer
booster.load_model(model, model_ckpt_path)
booster.load_optimizer(optimizer, optimizer_ckpt_path)

model.train()

# load data & init data
messages = []
with open(data_path) as f:
    for line in f:
        content = json.loads(line)['messages'][0]['content']
        messages.append(content)

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to("cuda")

generated_ids = model.generate(**model_inputs, max_new_tokens=512)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(f"response {response}")
