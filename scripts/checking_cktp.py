import torch

def inspect_ckpt_keys(ckpt_path, num_keys=10):
    print(f"Loading checkpoint from: {ckpt_path}\n")
    
    # Load the checkpoint (mapping to CPU prevents VRAM spikes)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    # Check if it's a valid Lightning checkpoint
    if "state_dict" not in checkpoint:
        print("Error: 'state_dict' not found. Are you sure this is a standard PyTorch Lightning .ckpt?")
        return
        
    state_dict = checkpoint["state_dict"]
    keys = list(state_dict.keys())
    
    print(f"Total tensor keys found: {len(keys)}")
    print(f"Showing the first {num_keys} keys to identify the prefix:\n")
    
    for i, key in enumerate(keys[:num_keys]):
        print(f"{i + 1}. {key}")

# --- How to use it ---
# Replace 'path/to/your/teacher_model.ckpt' with your actual file path
ckpt_file_path = "C:\\Users\\ikeru\\OneDrive\\Escritorio\\UNI IKER\\PROYECTOS\\End_2_End\\checkpoints\\last-v1.ckpt"
inspect_ckpt_keys(ckpt_file_path)