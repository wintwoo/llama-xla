import argparse
import json
import os
import re
import torch


EXACT_MATCH_WEIGHTS = [
    "lm_head.weight", 
    "model.embed_tokens.weight",
    "model.norm.weight",
]

DECODER_LAYER_REGEX = r"(model\.layers\.[0-9]*).*"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    return args

def reshard_and_save_weights(model_dir: str, output_dir: str):
    with open(os.path.join(model_dir, "pytorch_model.bin.index.json")) as f:
        index = json.load(f)

    ckpt_files = set([index["weight_map"][k] for k in index["weight_map"].keys()])
    print(f"Found these checkpoint files in pytorch_model_bin.index.json: {ckpt_files}")

    grouped_weights = {}
    p = re.compile(DECODER_LAYER_REGEX)

    for f in ckpt_files:
        print(f"Loading {f} ... ", end="", flush=True)
        ckpt = torch.load(os.path.join(model_dir, f))
        print("Done!")
        for weight in ckpt.keys():
            if weight in EXACT_MATCH_WEIGHTS:
                print(f"Saving {weight} ... ", end="", flush=True)
                weight_dict = {
                    weight: ckpt[weight]
                }
                torch.save(weight_dict, os.path.join(output_dir, f"{weight}.bin"))
                print("Done!")
            else:
                match = p.search(weight)
                if match and match.group(1):
                    block_name = match.group(1)
                    if block_name not in grouped_weights.keys():
                        grouped_weights[block_name] = {}
                    grouped_weights[block_name][weight] = ckpt[weight]
        
        for layer in grouped_weights.keys():
            print(f"Saving weights for layer {layer} ... ", end="", flush=True)
            torch.save(grouped_weights[layer], os.path.join(output_dir, f"{layer}.bin"))
            print("Done!")

def main():
    args = parse_args()
    reshard_and_save_weights(args.model, args.output)

if __name__ == "__main__":
    main()