import torch
from torchvision.utils import make_grid
from model_v2 import Generator
import os
from PIL import Image
import numpy as np

def slerp(low, high, val):
    """Spherical linear interpolation (SLERP) for stable norm traversal in latent space."""
    low_norm = low / (torch.norm(low, dim=-1, keepdim=True) + 1e-8)
    high_norm = high / (torch.norm(high, dim=-1, keepdim=True) + 1e-8)
    omega = torch.acos(torch.clamp(torch.sum(low_norm * high_norm, dim=-1, keepdim=True), -1, 1))
    so = torch.sin(omega)
    
    # Handle collinear vectors
    res = torch.where(
        so < 1e-5,
        (1.0 - val) * low + val * high,
        (torch.sin((1.0 - val) * omega) / so) * low + (torch.sin(val * omega) / so) * high
    )
    return res

def generate_interpolation_gif():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Initialize Generator matching the trained model parameters
    G = Generator(image_size=64, noise_dim=128).to(device)
    
    # Load weights
    ckpt_path = "epoch_0500.pt"
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found.")
        return
        
    print(f"Loading weights from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device)
    
    # Load base G weights and copy EMA
    G.load_state_dict(ckpt["G"], strict=False)
    if "ema" in ckpt:
        print("Applying EMA weights...")
        for name, param in G.named_parameters():
            if param.requires_grad and name in ckpt["ema"]:
                param.data.copy_(ckpt["ema"][name])
    
    # Disable unused attention layer
    G.attn_16.gamma.data.fill_(0.0)
    G.eval()

    # Create directories
    os.makedirs("generated_outputs", exist_ok=True)
    
    # 2. Define interpolation parameters
    grid_size = 16  # 4x4 grid
    num_steps = 60  # Frames per morph transition
    truncation = 0.75  # Limit extreme noise for clearer faces
    
    # Seed control for reproducibility
    torch.manual_seed(42)
    
    # Generate keyframe latent vectors (3 keyframes to make a loop: A -> B -> C -> A)
    z_a = torch.randn(grid_size, 128, device=device) * truncation
    z_b = torch.randn(grid_size, 128, device=device) * truncation
    z_c = torch.randn(grid_size, 128, device=device) * truncation
    
    keyframes = [z_a, z_b, z_c, z_a]
    frames = []

    print("Generating morphing frames...")
    with torch.no_grad():
        with torch.amp.autocast('cuda' if device == 'cuda' else 'cpu'):
            for i in range(len(keyframes) - 1):
                start_z = keyframes[i]
                end_z = keyframes[i+1]
                
                for step in range(num_steps):
                    # Progress coefficient (0.0 to 1.0)
                    alpha = step / num_steps
                    
                    # Smooth step easing for organic morph speed transitions
                    alpha_smooth = 3 * (alpha ** 2) - 2 * (alpha ** 3)
                    
                    # Spherical interpolate latents
                    z = slerp(start_z, end_z, alpha_smooth)
                    
                    # Generate and postprocess images
                    samples = G(z)
                    samples = samples.float().clamp(-1, 1)
                    samples = (samples + 1) / 2
                    
                    # Create grid
                    grid = make_grid(samples, nrow=4, padding=2)
                    
                    # Convert to PIL Image
                    grid_np = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                    img = Image.fromarray(grid_np)
                    frames.append(img)
                    
                    if (step + 1) % 20 == 0:
                        print(f"  Transition {i+1}/{len(keyframes)-1} — Step {step+1}/{num_steps}")

    # 3. Save as looping GIF
    gif_path = "generated_outputs/dog_morph_interpolation.gif"
    print(f"Saving frames to looping GIF: {gif_path}...")
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        optimize=True,
        duration=50,  # 50ms per frame = 20 fps
        loop=0
    )
    print("GIF saved successfully!")

if __name__ == "__main__":
    generate_interpolation_gif()
