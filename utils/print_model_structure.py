from safetensors import safe_open

def print_model_structure(model_path):
    # Load parameters from safetensors file
    safetensors_path = f"{model_path}/model.safetensors"
    print(f"Loading parameters from {safetensors_path}...")
    
    with safe_open(safetensors_path, framework="pt") as f:
        # Print all parameter names
        print("\nModel Parameters:")
        print("=" * 50)
        for key in f.keys():
            print(key)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    args = parser.parse_args()
    
    print_model_structure(args.model_path) 