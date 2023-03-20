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

n = tokenizer.encode('\n', return_tensors='pt')[0]

ctx = """A dialog, where User interacts with AI. AI is helpful, kind, obedient, honest, and knows its own limits.
User: Hello, AI.
AI: Hello! How can I assist you today?
"""

while True:
    print(ctx)
    prompt = input(f'User: ')
    if ctx != "":
        ctx = ctx + "User: " + prompt + "\n"
    else:
        ctx = prompt + "\n"

    ctx = (ctx[-1920:]) if len(ctx) >= 2048 else ctx

    if len(ctx.strip()) > 0:
        batch = tokenizer(ctx, return_tensors="pt")
        result = model.generate(batch["input_ids"].cpu(),
                                do_sample=True,
                                top_k=50,
                                max_length=2048,
                                top_p=0.95,
                                temperature=1.0,
                                eos_token_id=n
                                )
        decoded = tokenizer.decode(result[0])
        ctx = decoded + "\n"
