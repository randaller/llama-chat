import llamahf
import torch
import pandas as pd
from torch.utils.data import Dataset, random_split
from transformers import TrainingArguments, Trainer

MODEL = 'decapoda-research/llama-7b-hf'
DATA_FILE_PATH = 'datasets/elon_musk_tweets.csv'
OUTPUT_DIR = './trained'

texts = pd.read_csv(DATA_FILE_PATH)['text']

tokenizer = llamahf.LLaMATokenizer.from_pretrained(MODEL)
model = llamahf.LLaMAForCausalLM.from_pretrained(MODEL).cpu()

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))


class TextDataset(Dataset):
    def __init__(self, txt_list, tokenizer, max_length):
        self.labels = []
        self.input_ids = []
        self.attn_masks = []
        for txt in txt_list:
            encodings_dict = tokenizer(txt, truncation=True, max_length=max_length, padding="max_length")
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self): return len(self.input_ids)

    def __getitem__(self, idx): return self.input_ids[idx], self.attn_masks[idx]


dataset = TextDataset(texts, tokenizer, max_length=max([len(tokenizer.encode(text)) for text in texts]))
train_dataset, val_dataset = random_split(dataset, [int(0.9 * len(dataset)), len(dataset) - int(0.9 * len(dataset))])

training_args = TrainingArguments(
    save_steps=5000,
    warmup_steps=10,
    logging_steps=100,
    weight_decay=0.05,
    num_train_epochs=1,
    logging_dir='./logs',
    output_dir=OUTPUT_DIR,
    no_cuda=True,
    # bf16=True,
    per_device_eval_batch_size=1,
    per_device_train_batch_size=1)

trainer = Trainer(model=model,
                  args=training_args,
                  eval_dataset=val_dataset,
                  train_dataset=train_dataset,
                  data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                              'attention_mask': torch.stack([f[1] for f in data]),
                                              'labels': torch.stack([f[0] for f in data])})

trainer.train()
trainer.save_model()
tokenizer.save_pretrained(OUTPUT_DIR)
del trainer

sample_outputs = model.generate(tokenizer('', return_tensors="pt").input_ids.cpu(),
                                do_sample=True,
                                top_k=50,
                                max_length=300,
                                top_p=0.95,
                                temperature=1.0)

print(tokenizer.decode(sample_outputs[0]))
