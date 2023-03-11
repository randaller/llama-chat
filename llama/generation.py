# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

# Copyright by Shawn Presser
# https://github.com/shawwn/
# taken here
# https://github.com/shawwn/llama/commit/40d99d329a5e38d85904d3a6519c54e6dd6ee9e1

from typing import List

import torch
import traceback

from llama.tokenizer import Tokenizer
from llama.model import Transformer
from tqdm import trange


class LLaMA:
    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(
            self,
            prompts: List[str],
            max_gen_len: int,
            temperature: float = 0.8,
            top_p: float = 0.95,
            top_k: int = 40,
            repetition_penalty: float = (1.0 / 0.85),
            sampler: str = 'top_k',
    ) -> List[str]:
        bsz = len(prompts)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        count_newlines = prompts[0].count("\n")

        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
            tokens[k, -1] = self.tokenizer.eos_id
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        decoded = [None] * bsz

        for cur_pos in trange(start_pos, total_len, desc="forward"):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)

            # repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
            if repetition_penalty != 1.0:
                logits_new = logits.clone()
                batch_size = len(tokens)
                for i in range(batch_size):
                    for token in set(tokens[i].tolist()):
                        # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                        if logits[i, token] < 0:
                            logits_new[i, token] = logits[i, token] * repetition_penalty
                        else:
                            logits_new[i, token] = logits[i, token] / repetition_penalty
                logits = logits_new

            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                if sampler == 'top_k':
                    next_token = sample_top_k(probs, top_p=top_p, top_k=top_k)
                else:
                    next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1).cpu()
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos

            print("-" * 30)
            for i, t in enumerate(tokens.tolist()):
                # i = cur_pos
                # t = next_token
                # cut to max gen len
                # t = t[: len(pr-ompt_tokens[i]) + max_gen_len]
                t = t[: min(cur_pos, len(prompt_tokens[i]) + max_gen_len)]
                # cut to eos tok if any
                try:
                    t = t[: t.index(self.tokenizer.eos_id)]
                except ValueError:
                    pass  # traceback.print_exc()
                try:
                    d = self.tokenizer.decode(t)
                    print(d)
                    decoded[i] = d
                    
                    result_count_newlines = d.count("\n")
                    if result_count_newlines > count_newlines:
                        return decoded

                except IndexError:
                    traceback.print_exc()
                    print(t)
            print("-" * 30)
        return decoded


# default sampler
def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


# sampler by Shawn
def sample_top_k(probs, top_p=0.0, top_k=40):
    if top_k > 0:
        probs_sort, probs_idx = torch.topk(probs, top_k)
    else:
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    if top_p > 0.0:
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > top_p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
