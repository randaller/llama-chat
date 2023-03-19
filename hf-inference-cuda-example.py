import llamahf
from accelerate import infer_auto_device_map

# # to save memory use bfloat16 on cpu
# import torch
# torch.set_default_dtype(torch.bfloat16)

MODEL = 'decapoda-research/llama-7b-hf'
# MODEL = 'decapoda-research/llama-13b-hf'
# MODEL = 'decapoda-research/llama-30b-hf'
# MODEL = 'decapoda-research/llama-65b-hf'

# MODEL = './trained'

tokenizer = llamahf.LLaMATokenizer.from_pretrained(MODEL)
model = llamahf.LLaMAForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, device_map="auto", offload_folder="./offload")

# will use 6 Gb of GPU VRAM, others to CPU RAM
device_map = infer_auto_device_map(model, max_memory={0: "6GiB", "cpu": "128GiB"})
print(device_map)

batch = tokenizer("The highest mountain in China is ", return_tensors="pt")
print(tokenizer.decode(model.generate(batch["input_ids"].cuda(), max_length=100)[0]))
