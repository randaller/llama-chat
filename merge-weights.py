# Original copyright by Jason Phang
# https://github.com/zphang
# Taken here
# https://github.com/huggingface/transformers/pull/21955/commits/8978f28e6c44b083c0b190d3931902c2904c940a#diff-110a445233a8b15a0875998eeaf75cb8607b38a5daa736291dd058766879bbdd

import argparse
import json
import os
import shutil
import torch

"""
Sample usage:
    ```
    python merge_weights.py --input_dir D:\Downloads\LLaMA --model_size 13B
    ```
"""

INTERMEDIATE_SIZE_MAP = {
    "7B": 11008,
    "13B": 13824,
    "30B": 17920,
    "65B": 22016,
}

NUM_SHARDS = {
    "7B": 1,
    "13B": 2,
    "30B": 4,
    "65B": 8,
}


def read_json(path):
    with open(path, "r") as f:
        return json.loads(f.read())


def write_model(input_base_path, model_size):
    assert model_size in INTERMEDIATE_SIZE_MAP

    params = read_json(os.path.join(input_base_path, "params.json"))
    num_shards = NUM_SHARDS[model_size]
    n_layers = params["n_layers"]
    n_heads = params["n_heads"]
    n_heads_per_shard = n_heads // num_shards
    dim = params["dim"]
    dims_per_head = dim // n_heads

    # Load weights
    if model_size == "7B":
        loaded = torch.load(os.path.join(input_base_path, "consolidated.00.pth"), map_location="cpu")
    else:
        loaded = [
            torch.load(os.path.join(input_base_path, f"consolidated.{i:02d}.pth"), map_location="cpu")
            for i in range(num_shards)
        ]

    state_dict = {}

    for layer_i in range(n_layers):
        if model_size == "7B":
            state_dict |= {
                f"layers.{layer_i}.attention.wq.weight": loaded[
                    f"layers.{layer_i}.attention.wq.weight"
                ],
                f"layers.{layer_i}.attention.wk.weight": loaded[
                    f"layers.{layer_i}.attention.wk.weight"
                ],
                f"layers.{layer_i}.attention.wv.weight": loaded[
                    f"layers.{layer_i}.attention.wv.weight"
                ],
                f"layers.{layer_i}.attention.wo.weight": loaded[
                    f"layers.{layer_i}.attention.wo.weight"
                ],
                f"layers.{layer_i}.feed_forward.w1.weight": loaded[
                    f"layers.{layer_i}.feed_forward.w1.weight"
                ],
                f"layers.{layer_i}.feed_forward.w2.weight": loaded[
                    f"layers.{layer_i}.feed_forward.w2.weight"
                ],
                f"layers.{layer_i}.feed_forward.w3.weight": loaded[
                    f"layers.{layer_i}.feed_forward.w3.weight"
                ],
                f"layers.{layer_i}.attention_norm.weight": loaded[
                    f"layers.{layer_i}.attention_norm.weight"
                ],
                f"layers.{layer_i}.ffn_norm.weight": loaded[f"layers.{layer_i}.ffn_norm.weight"],
            }
        else:
            state_dict |= {
                f"layers.{layer_i}.attention_norm.weight": loaded[0][
                    f"layers.{layer_i}.attention_norm.weight"
                ],
                f"layers.{layer_i}.ffn_norm.weight": loaded[0][f"layers.{layer_i}.ffn_norm.weight"],
            }
            state_dict[f"layers.{layer_i}.attention.wq.weight"] = torch.cat(
                [
                    loaded[i][f"layers.{layer_i}.attention.wq.weight"].view(n_heads_per_shard, dims_per_head, dim)
                    for i in range(num_shards)
                ],
                dim=0,
            ).reshape(dim, dim)
            state_dict[f"layers.{layer_i}.attention.wk.weight"] = torch.cat(
                [
                    loaded[i][f"layers.{layer_i}.attention.wk.weight"].view(n_heads_per_shard, dims_per_head, dim)
                    for i in range(num_shards)
                ],
                dim=0,
            ).reshape(dim, dim)
            state_dict[f"layers.{layer_i}.attention.wv.weight"] = torch.cat(
                [
                    loaded[i][f"layers.{layer_i}.attention.wv.weight"].view(n_heads_per_shard, dims_per_head, dim)
                    for i in range(num_shards)
                ],
                dim=0,
            ).reshape(dim, dim)
            state_dict[f"layers.{layer_i}.attention.wo.weight"] = torch.cat(
                [loaded[i][f"layers.{layer_i}.attention.wo.weight"] for i in range(num_shards)], dim=1
            )
            state_dict[f"layers.{layer_i}.feed_forward.w1.weight"] = torch.cat(
                [loaded[i][f"layers.{layer_i}.feed_forward.w1.weight"] for i in range(num_shards)], dim=0
            )
            state_dict[f"layers.{layer_i}.feed_forward.w2.weight"] = torch.cat(
                [loaded[i][f"layers.{layer_i}.feed_forward.w2.weight"] for i in range(num_shards)], dim=1
            )
            state_dict[f"layers.{layer_i}.feed_forward.w3.weight"] = torch.cat(
                [loaded[i][f"layers.{layer_i}.feed_forward.w3.weight"] for i in range(num_shards)], dim=0
            )

    if model_size == "7B":
        state_dict |= {
            "tok_embeddings.weight": loaded["tok_embeddings.weight"],
            "norm.weight": loaded["norm.weight"],
            "output.weight": loaded["output.weight"],
        }
    else:
        state_dict |= {
            "norm.weight": loaded[0]["norm.weight"],
            "tok_embeddings.weight": torch.cat(
                [loaded[i]["tok_embeddings.weight"] for i in range(num_shards)], dim=1
            ),
            "output.weight": torch.cat([loaded[i]["output.weight"] for i in range(num_shards)], dim=0),
        }

    torch.save(state_dict, 'merged.pth')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        help="Location of LLaMA weights, which contains tokenizer.model and model folders",
    )
    parser.add_argument(
        "--model_size",
        choices=["7B", "13B", "30B", "65B"],
    )
    args = parser.parse_args()

    write_model(
        input_base_path=os.path.join(args.input_dir, args.model_size),
        model_size=args.model_size,
    )


if __name__ == "__main__":
    main()
