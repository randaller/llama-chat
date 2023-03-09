# Chat with Meta's LLaMA models at home made easy

This repository is a chat example with [LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) ([arXiv](https://arxiv.org/abs/2302.13971v1)) models running on a typical home PC. You will just need a NVIDIA videocard and some RAM to chat with model.

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

