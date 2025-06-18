#!/usr/bin/env python3

from safetensors.torch import load_file
import torch

def check_safetensor_params():
    safetensor_path = "/sgl-workspace/sgl-tune-eagle/llama4_draft_freeze/model.safetensors"
    try:
        state_dict = load_file(safetensor_path)
        print("Safetensor parameters:")
        print("=" * 50)
        for key in sorted(state_dict.keys()):
            print(f"  {key}")
        print(f"\nTotal parameters: {len(state_dict)}")
    except Exception as e:
        print(f"Error loading safetensor file: {e}")

if __name__ == "__main__":
    check_safetensor_params() 