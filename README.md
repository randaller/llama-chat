# Chat with Meta's LLaMA models at home made easy

This repository is a chat example with [LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) ([arXiv](https://arxiv.org/abs/2302.13971v1)) models running on a typical home PC. You will just need a NVIDIA videocard and some RAM to chat with model.

This repo is heavily based on Meta's original repo: https://github.com/facebookresearch/llama

And on Venuatu's repo: https://github.com/venuatu/llama

### Examples of chats here

https://github.com/facebookresearch/llama/issues/162

### System requirements
- Modern enough CPU
- NVIDIA graphics card
- 64 or better 128 Gb of RAM (192 or 256 would be perfect)

One may run with 32 Gb of RAM, but inference will be slow (with the speed of your swap file reading)

I am running this on 12700k/128 Gb RAM/NVIDIA 3070ti 8Gb/fast huge nvme and getting one token from 30B model in a few seconds.

### Conda Environment Setup Example for Windows 10+
Download and install Anaconda Python https://www.anaconda.com and run Anaconda Prompt
```
conda create -n llama python=3.10
conda activate llama
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

### Setup
In a conda env with pytorch / cuda available, run
```
pip install -r requirements.txt
```
Then in this repository
```
pip install -e .
```

### Download tokenizer and models
magnet:?xt=urn:btih:ZXXDAUWYLRUXXBHUYEMS6Q5CE5WA3LVA&dn=LLaMA

or

magnet:xt=urn:btih:b8287ebfa04f879b048d4d4404108cf3e8014352&dn=LLaMA&tr=udp%3a%2f%2ftracker.opentrackr.org%3a1337%2fannounce

### Prepare model

First, you need to unshard model checkpoints to a single file. Let's do this for 30B model.

```
python merge_weights.py --input_dir D:\Downloads\LLaMA --model_size 30B
```

In this example, D:\Downloads\LLaMA is a root folder of downloaded torrent with weights.

This will create merged.pth file in the root folder of this repo.

Place this file and corresponding (torrentroot)/30B/params.json of model into [/model] folder.

So you should end up with two files in [/model] folder: merged.pth and params.json.

Place (torrentroot)/tokenizer.model file to the [/tokenizer] folder of this repo. Now you are ready to go.

### Run the chat

```
python example-chat.py ./model ./tokenizer/tokenizer.model
```
