import llamahf
import os

# # to save memory use bfloat16
# import torch
# torch.set_default_dtype(torch.bfloat16)

MODEL = 'decapoda-research/llama-7b-hf'
# MODEL = 'decapoda-research/llama-13b-hf'
# MODEL = 'decapoda-research/llama-30b-hf'
# MODEL = 'decapoda-research/llama-65b-hf'

if os.path.exists('./trained'):
    MODEL = './trained'

tokenizer = llamahf.LLaMATokenizer.from_pretrained(MODEL)
model = llamahf.LLaMAForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True)
model.to('cpu')

batch = tokenizer("The highest mountain in China is ", return_tensors="pt")
print(tokenizer.decode(model.generate(batch["input_ids"].cpu(), do_sample=True, top_k=50, max_length=100, top_p=0.95, temperature=1.0)[0]))
