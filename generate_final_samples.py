import torch
from torchvision.utils import save_image, make_grid
from model_v2 import Generator
import os

def generate_samples():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize Generator matching args used: image_size=64, noise_dim=128
    G = Generator(image_size=64, noise_dim=128).to(device)
    
    # Load weights
    ckpt_path = "epoch_0500.pt"
    print(f"Loading weights from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device)
    
    # 1. Load G state dict (which contains correct BN running statistics and base weights)
    # Since G has attn_16 but the checkpoint does not, we load with strict=False
    G.load_state_dict(ckpt["G"], strict=False)
    
    # 2. Copy EMA parameters on top of the loaded model
    if "ema" in ckpt:
        print("Applying EMA weights...")
        for name, param in G.named_parameters():
            if param.requires_grad and name in ckpt["ema"]:
                param.data.copy_(ckpt["ema"][name])
    else:
        print("No EMA weights found, using standard weights.")

    # 3. Explicitly disable attn_16 by setting its gamma to 0
    # since it was not present during training of this checkpoint
    G.attn_16.gamma.data.fill_(0.0)
        
    G.eval()

    os.makedirs("generated_outputs", exist_ok=True)
    
    print("Generating images...")
    with torch.no_grad():
        with torch.amp.autocast('cuda' if device == 'cuda' else 'cpu'):
            # Generate 64 sample variations
            z = torch.randn(64, 128, device=device)
            
            # Truncation trick for clearer images (0.8 is standard)
            z = z * 0.8
            
            samples = G(z)
            samples = samples.float().clamp(-1, 1)
            samples = (samples + 1) / 2
            
            grid = make_grid(samples, nrow=8, padding=2)
            out_path = "generated_outputs/epoch_0500_samples.png"
            save_image(grid, out_path)
            print(f"Saved grid to: {out_path}")

if __name__ == "__main__":
    generate_samples()
