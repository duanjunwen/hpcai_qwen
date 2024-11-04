import json
import torch
import colossalai
from tqdm import tqdm
from colossalai.booster import Booster
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer.hybrid_adam import HybridAdam
from transformers import AutoModelForCausalLM, AutoTokenizer
from colossalai.booster.plugin import HybridParallelPlugin

# model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model_name = "/root/dataDisk/model/qwen"
data_path = "/root/commonData/Wukong/wukong.jsonl"

# init dist env
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

# init model, tokenizer, optimizer
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="cuda"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
optimizer = HybridAdam(model.parameters())

# load data & init data
messages = []
with open(data_path) as f:
    for line in f:
        content = json.loads(line)['messages'][0]['content']
        messages.append(content)

encoded_batch = tokenizer(messages, padding=True, truncation=True, return_tensors="pt")
dataloader = plugin.prepare_dataloader(encoded_batch, batch_size=4, shuffle=True, drop_last=True, seed=42)
model, optimizer, _, dataloader, _ = booster.boost(model, optimizer, dataloader=dataloader)

# Train 
for epoch in range(10):
    # for step, batch in enumerate(tqdm(iter([encoded_batch]), desc="Step")):
    for step, batch in enumerate(tqdm(dataloader, desc="Step")):
        for k, v in batch.items():
            batch[k] = v.to('cuda:0')
        outputs = model(**batch)
        loss = outputs[0]
        del outputs  # free memory
        print(f"Epoch {epoch} Step {step} loss: {loss}")
        # loss.mean().backward()
        booster.backward(loss, optimizer)
        optimizer.step()
        optimizer.zero_grad()

# save model
model_ckpt_path = "/root/dataDisk/model/qwen_save"
optimizer_ckpt_path = "/root/dataDisk/model/qwen_optim_save"
booster.save_model(model, model_ckpt_path)
booster.save_optimizer(optimizer, optimizer_ckpt_path)