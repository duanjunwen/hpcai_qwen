import json
from transformers import AutoModelForCausalLM, AutoTokenizer

# model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model_name = "/root/dataDisk/model/qwen"
model_ckpt_path = "/root/dataDisk/model/qwen_save"
data_path = "/root/commonData/Wukong/wukong.jsonl"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

with open(data_path) as f:
    for line in f:
        content = json.loads(line)['messages'][0]
        new_content = {"role": content['from'], "content": content['content']}
        messages.append(new_content)
print(f"messages {messages}")

for message in messages[:5]:
    text = tokenizer.apply_chat_template(
        [message],
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(**model_inputs, max_new_tokens=512)
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"response {response}")