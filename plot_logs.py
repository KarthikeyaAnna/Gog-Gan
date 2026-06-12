import json
import os
import matplotlib
# Use headless backend to prevent X11 display connection / GUI segfaults
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Set style for clean, premium-looking plots
plt.rcParams['figure.facecolor'] = '#ffffff'
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['axes.edgecolor'] = '#dee2e6'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.color'] = '#e9ecef'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.8
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.labelcolor'] = '#212529'
plt.rcParams['xtick.color'] = '#495057'
plt.rcParams['ytick.color'] = '#495057'
plt.rcParams['text.color'] = '#212529'

def plot_training_metrics():
    log_path = "Log.json"
    if not os.path.exists(log_path):
        print(f"Error: {log_path} not found.")
        return
        
    with open(log_path, 'r') as f:
        data = json.load(f)
        
    epochs = [x['epoch'] for x in data]
    d_loss = [x['d_loss'] for x in data]
    g_loss = [x['g_loss'] for x in data]
    r1 = [x['r1'] for x in data]
    time_taken = [x['time'] for x in data]
    ram_gb = [x['ram_gb'] for x in data]
    vram_gb = [x['vram_gb'] for x in data]
    
    os.makedirs("generated_outputs/plots", exist_ok=True)
    
    # 1. Loss Plot (Adversarial Relationship)
    plt.figure(figsize=(10, 5), dpi=300)
    plt.plot(epochs, d_loss, color='#e63946', label='Discriminator Loss (D)', alpha=0.8, linewidth=1.5)
    plt.plot(epochs, g_loss, color='#457b9d', label='Generator Loss (G)', alpha=0.8, linewidth=1.5)
    plt.title('GAN Loss Curves (D and G)', fontsize=14, fontweight='bold', pad=15, color='#1d3557')
    plt.xlabel('Epoch', fontsize=11, labelpad=8)
    plt.ylabel('Loss', fontsize=11, labelpad=8)
    plt.legend(frameon=True, facecolor='#ffffff', edgecolor='#dee2e6', framealpha=0.9, loc='upper right')
    plt.tight_layout()
    plt.savefig('generated_outputs/plots/loss_curves.png', bbox_inches='tight')
    plt.close()
    
    # 2. R1 Gradient Penalty Plot (Log Scale for readability)
    plt.figure(figsize=(10, 5), dpi=300)
    plt.plot(epochs, r1, color='#8f2d56', linewidth=1.5, label='R1 Regularization Penalty')
    plt.yscale('log')
    plt.title('Discriminator R1 Gradient Penalty (Log Scale)', fontsize=14, fontweight='bold', pad=15, color='#1d3557')
    plt.xlabel('Epoch', fontsize=11, labelpad=8)
    plt.ylabel('R1 Penalty Value', fontsize=11, labelpad=8)
    plt.legend(frameon=True, facecolor='#ffffff', edgecolor='#dee2e6', framealpha=0.9, loc='upper right')
    plt.tight_layout()
    plt.savefig('generated_outputs/plots/r1_penalty.png', bbox_inches='tight')
    plt.close()
    
    # 3. Epoch Training Time Plot
    plt.figure(figsize=(10, 5), dpi=300)
    plt.plot(epochs, time_taken, color='#2a9d8f', alpha=0.3, label='Epoch Duration (raw)')
    
    # Calculate 10-epoch moving average
    window_size = 10
    rolling_time = np.convolve(time_taken, np.ones(window_size)/window_size, mode='valid')
    rolling_epochs = epochs[window_size-1:]
    plt.plot(rolling_epochs, rolling_time, color='#264653', linewidth=2, label=f'{window_size}-Epoch Moving Avg')
    
    plt.title('Training Duration per Epoch', fontsize=14, fontweight='bold', pad=15, color='#1d3557')
    plt.xlabel('Epoch', fontsize=11, labelpad=8)
    plt.ylabel('Time (seconds)', fontsize=11, labelpad=8)
    plt.legend(frameon=True, facecolor='#ffffff', edgecolor='#dee2e6', framealpha=0.9, loc='upper right')
    plt.tight_layout()
    plt.savefig('generated_outputs/plots/epoch_time.png', bbox_inches='tight')
    plt.close()
    
    # 4. Memory Usage (RAM and VRAM) Plot
    plt.figure(figsize=(10, 5), dpi=300)
    plt.plot(epochs, ram_gb, color='#f4a261', linewidth=1.8, label='CPU RAM (GB)')
    plt.plot(epochs, vram_gb, color='#e76f51', linewidth=1.8, label='GPU VRAM (GB)')
    plt.title('Memory Utilization Profile', fontsize=14, fontweight='bold', pad=15, color='#1d3557')
    plt.xlabel('Epoch', fontsize=11, labelpad=8)
    plt.ylabel('Memory Usage (GB)', fontsize=11, labelpad=8)
    plt.legend(frameon=True, facecolor='#ffffff', edgecolor='#dee2e6', framealpha=0.9, loc='center right')
    plt.tight_layout()
    plt.savefig('generated_outputs/plots/memory_usage.png', bbox_inches='tight')
    plt.close()
    
    print("All plots generated successfully under 'generated_outputs/plots/'.")

if __name__ == "__main__":
    plot_training_metrics()
