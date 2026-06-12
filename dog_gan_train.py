"""
Dog-GAN Training Script

Trains an unconditional GAN on Stanford Dogs dataset.
Uses hinge loss, R1 gradient penalty, and exponential moving average.

Optimized for RTX 2050 (4GB VRAM):
  - Mixed precision training (FP16)
  - Gradient accumulation
  - No CLIP encoder overhead
"""
# Auto-install dependencies on Colab VM
import subprocess
import sys

try:
    import datasets
except ImportError:
    print("[+] Installing dependency datasets...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])

import os
import sys
import time
import argparse
import json
import datetime
import psutil
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import torch
# Enable TensorFloat-32 (TF32) for RTX 40/50 series tensor cores speedup
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn as nn
import torch.nn.functional as F

from torchvision.utils import save_image, make_grid

try:
    from model import Generator, Discriminator
except ImportError:
    Generator, Discriminator = None, None
try:
    from model_v2 import Generator as GeneratorV2, Discriminator as DiscriminatorV2
except ImportError:
    GeneratorV2 = None
    DiscriminatorV2 = None
from diffaugment import DiffAugment
from dataset import AFHQDogDataset, download_stanford_dogs, get_dataloader


def hinge_loss_dis(real_score, fake_score):
    """Hinge loss for discriminator."""
    return (F.relu(1.0 - real_score) + F.relu(1.0 + fake_score)).mean()


def hinge_loss_gen(fake_score):
    """Hinge loss for generator."""
    return -fake_score.mean()


def r1_penalty(real_images, real_score):
    """R1 gradient penalty for discriminator regularization."""
    grad = torch.autograd.grad(
        outputs=real_score.sum(),
        inputs=real_images,
        create_graph=True,
    )[0]
    return grad.pow(2).reshape(grad.size(0), -1).sum(1).mean()


def sanitize_bn_buffers(model):
    """
    Prevents float16 AMP overflows from permanently corrupting BatchNorm running stats.
    Scans for NaN/Inf in running_mean and running_var, resetting them if corrupted.
    """
    for name, buf in model.named_buffers():
        if "running_mean" in name or "running_var" in name:
            if torch.isnan(buf).any() or torch.isinf(buf).any():
                if "running_mean" in name:
                    buf.zero_()
                else:
                    buf.fill_(1.0)


def get_raw_module(model):
    """Recursively unwrap any DDP or torch.compile wrappers to get the raw nn.Module."""
    m = model
    while hasattr(m, '_orig_mod') or hasattr(m, 'module'):
        if hasattr(m, '_orig_mod'):
            m = m._orig_mod
        if hasattr(m, 'module'):
            m = m.module
    return m


class EMA:
    """Exponential Moving Average for generator weights."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])


class DriveSync:
    """Utility to sync training checkpoints to Google Drive for preemption resilience."""

    def __init__(self, token_path="token.json"):
        self.token_path = token_path
        self.service = None
        self.folder_id = None
        self.scopes = ['https://www.googleapis.com/auth/drive.file']
        
        if os.path.exists(token_path):
            try:
                from google.oauth2.credentials import Credentials
                from googleapiclient.discovery import build
                creds = Credentials.from_authorized_user_file(token_path, self.scopes)
                self.service = build('drive', 'v3', credentials=creds)
                print("[DriveSync] Successfully authenticated with Google Drive.")
                self.folder_id = self._get_or_create_folder("Dog-GAN-Checkpoints")
            except Exception as e:
                print(f"[DriveSync] Error authenticating with Drive: {e}")

    def _get_or_create_folder(self, folder_name):
        try:
            query = f"name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
            results = self.service.files().list(q=query, fields="files(id)").execute()
            files = results.get('files', [])
            if files:
                return files[0]['id']
            
            folder_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            folder = self.service.files().create(body=folder_metadata, fields='id').execute()
            return folder.get('id')
        except Exception as e:
            print(f"[DriveSync] Error getting/creating folder: {e}")
            return None

    def upload_file(self, local_path, remote_name):
        if not self.service or not self.folder_id:
            return
        try:
            from googleapiclient.http import MediaFileUpload
            
            query = f"name = '{remote_name}' and '{self.folder_id}' in parents and trashed = false"
            results = self.service.files().list(q=query, fields="files(id)").execute()
            files = results.get('files', [])
            
            # Kaggle has high latency; using a huge chunk size (100MB) dramatically speeds up uploads
            media = MediaFileUpload(local_path, resumable=True, chunksize=100 * 1024 * 1024)
            if files:
                file_id = files[0]['id']
                self.service.files().update(fileId=file_id, media_body=media).execute()
                print(f"[DriveSync] Updated existing {remote_name} in Google Drive.")
            else:
                file_metadata = {
                    'name': remote_name,
                    'parents': [self.folder_id]
                }
                self.service.files().create(body=file_metadata, media_body=media).execute()
                print(f"[DriveSync] Uploaded new {remote_name} to Google Drive.")
        except Exception as e:
            print(f"[DriveSync] Error uploading file: {e}")

    def download_file(self, remote_name, local_path):
        if not self.service or not self.folder_id:
            return False
        try:
            from googleapiclient.http import MediaIoBaseDownload
            import io
            
            query = f"name = '{remote_name}' and '{self.folder_id}' in parents and trashed = false"
            results = self.service.files().list(q=query, fields="files(id)").execute()
            files = results.get('files', [])
            if not files:
                print(f"[DriveSync] {remote_name} not found in Google Drive.")
                return False
            
            file_id = files[0]['id']
            request = self.service.files().get_media(fileId=file_id)
            
            os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
            with open(local_path, "wb") as f:
                downloader = MediaIoBaseDownload(f, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
            print(f"[DriveSync] Successfully downloaded {remote_name} to local path.")
            return True
        except Exception as e:
            print(f"[DriveSync] Error downloading file: {e}")
            return False

    def check_and_delete_file(self, remote_name):
        if not self.service or not self.folder_id:
            return False
        try:
            query = f"name = '{remote_name}' and '{self.folder_id}' in parents and trashed = false"
            results = self.service.files().list(q=query, fields="files(id)").execute()
            files = results.get('files', [])
            if files:
                file_id = files[0]['id']
                self.service.files().delete(fileId=file_id).execute()
                print(f"[DriveSync] Detected stop signal ({remote_name}).")
                return True
        except Exception as e:
            print(f"[DriveSync] Error checking/deleting stop signal: {e}")
        return False


def train_worker(gpu, args):
    if args.multi_gpu and torch.cuda.device_count() > 1:
        dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:23456', world_size=torch.cuda.device_count(), rank=gpu, timeout=datetime.timedelta(minutes=30))
        device = torch.device(f"cuda:{gpu}")
        torch.cuda.set_device(gpu)
        is_master = (gpu == 0)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_master = True

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        
    if is_master:
        print(f"\n{'='*60}")
        print(f"  🐕 Dog-GAN Training")
        print(f"  Device: {device} (Multi-GPU: {args.multi_gpu})")
        print(f"  Image size: {args.image_size}×{args.image_size}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Epochs: {args.epochs}")
        print(f"{'='*60}\n")
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        os.makedirs(args.sample_dir, exist_ok=True)

    # Initialize Drive Sync helper and load checkpoint if available (Only Master)
    drive_sync = DriveSync() if is_master else None
    latest_ckpt_name = f"checkpoint_{args.session}.pt"
    log_name = f"training_log_{args.session}.json"
    latest_ckpt_path = os.path.join(args.checkpoint_dir, latest_ckpt_name)
    log_path = os.path.join(args.checkpoint_dir, log_name)
    
    if is_master:
        if not os.path.exists(latest_ckpt_path):
            print(f"[Train] Checking Google Drive for existing checkpoint ({latest_ckpt_name})...")
            drive_sync.download_file(latest_ckpt_name, latest_ckpt_path)
        if not os.path.exists(log_path):
            print(f"[Train] Checking Google Drive for existing training log ({log_name})...")
            drive_sync.download_file(log_name, log_path)

    if args.multi_gpu and torch.cuda.device_count() > 1:
        dist.barrier() # wait for master to download

    # ---- Dataset & DataLoader ----
    if args.download:
        if is_master:
            download_stanford_dogs(args.data_dir)
        if args.multi_gpu and torch.cuda.device_count() > 1:
            dist.barrier()

    dataset = AFHQDogDataset(root_dir=args.data_dir, image_size=args.image_size)
    
    use_ddp = getattr(args, 'multi_gpu', False) and torch.cuda.device_count() > 1
    sampler = DistributedSampler(dataset, shuffle=True) if use_ddp else None
    per_gpu_batch = args.batch_size // torch.cuda.device_count() if use_ddp else args.batch_size
    
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=per_gpu_batch,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    if is_master:
        print(f"[Dataset] Loaded {len(dataset)} images.")
        print(f"[Train] Dataloader ready — {len(dataloader)} batches/epoch (batch_size={per_gpu_batch} per GPU)\n")

    # ---- Models ----
    if getattr(args, 'model_v2', False):
        if GeneratorV2 is None:
            raise RuntimeError("model_v2.py not found! Cannot use --model_v2 flag.")
        if is_master: print("[Train] Using V2 architecture (high-capacity, ~24M params)")
        G = GeneratorV2(image_size=args.image_size, noise_dim=args.noise_dim).to(device)
        D = DiscriminatorV2(image_size=args.image_size).to(device)
    else:
        G = Generator(image_size=args.image_size, noise_dim=args.noise_dim).to(device)
        D = Discriminator(image_size=args.image_size).to(device)

    # Note: We CANNOT use torch.compile. It fundamentally breaks Exponential Moving Average (EMA) 
    # parameter swapping via .copy_(), and conflicts with AMP autocast inside the Generator, 
    # leading to catastrophic NaNs and the solid green image mode-collapse bug.
    # Multi-GPU support (DistributedDataParallel)
    if args.multi_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        if is_master: print(f"[Train] Enabling Multi-GPU DistributedDataParallel (DDP) for maximum performance!")
        G = DDP(G, device_ids=[gpu], find_unused_parameters=False)
        # We do NOT wrap D in DDP. DDP hooks are fundamentally incompatible with 
        # the R1 penalty's `create_graph=True` and multiple backwards passes.
        # We will manually synchronize D's gradients via dist.all_reduce() instead.

    # Get underlying base models for clean saving/loading/EMA
    base_G = G.module if hasattr(G, 'module') else G
    base_D = D.module if hasattr(D, 'module') else D

    # Compile Generator with torch.compile() for massive RTX 40/50 series speedup
    if hasattr(torch, "compile"):
        if is_master: print("[Train] Compiling Generator with torch.compile() for speedup...")
        if hasattr(G, "module"):
            G.module = torch.compile(G.module)
        else:
            G = torch.compile(G)

    if is_master:
        g_params = sum(p.numel() for p in base_G.parameters())
        d_params = sum(p.numel() for p in base_D.parameters())
        print(f"[Train] Generator:     {g_params:>10,} params")
        print(f"[Train] Discriminator: {d_params:>10,} params")
        print(f"[Train] Total:         {g_params + d_params:>10,} params\n")

    # ---- Optimizers ----
    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.0, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.0, 0.999))

    # ---- Mixed Precision ----
    # G uses standard AMP. D uses AMP for its main forward pass to save VRAM,
    # but computes R1 penalty in float32 to prevent second-order gradient overflow.
    scaler_G = torch.amp.GradScaler('cuda')
    scaler_D = torch.amp.GradScaler('cuda')
    g_scaler_skips = 0
    d_scaler_skips = 0

    # ---- EMA ----
    ema = EMA(get_raw_module(base_G), decay=args.ema_decay)

    # ---- Fixed noise for consistent samples ----
    fixed_noise = torch.randn(16, args.noise_dim, device=device)

    # ---- Resume from checkpoint ----
    start_epoch = 0
    if os.path.exists(latest_ckpt_path) and is_master and not getattr(args, 'fresh', False):
        print(f"[Train] Found checkpoint: {latest_ckpt_path}. Resuming...")
        try:
            ckpt = torch.load(latest_ckpt_path, map_location=device)
            get_raw_module(base_G).load_state_dict(ckpt["G"])
            get_raw_module(base_D).load_state_dict(ckpt["D"])
            opt_G.load_state_dict(ckpt["opt_G"])
            opt_D.load_state_dict(ckpt["opt_D"])
            # Override learning rates with command line arguments on resume
            for g in opt_G.param_groups: g['lr'] = args.lr_g
            for g in opt_D.param_groups: g['lr'] = args.lr_d
            start_epoch = ckpt["epoch"] + 1
            if "ema" in ckpt:
                ema.shadow = ckpt["ema"]
            if is_master: print(f"[Train] Successfully resumed from epoch {start_epoch}")
        except Exception as e:
            if is_master: print(f"[-] Failed to load checkpoint: {e}. Starting from scratch.")
    elif getattr(args, 'fresh', False) and is_master:
        print(f"[Train] --fresh flag set: training from scratch (ignoring any existing checkpoint).")

    # ---- Training log ----
    training_log = []
    if os.path.exists(log_path):
        try:
            with open(log_path, "r") as f:
                raw_log = json.load(f)
            
            # Clean up potentially concatenated logs from previous bugs
            valid_log = []
            for entry in raw_log:
                if entry.get("epoch") == 1:
                    valid_log = [entry]  # Reset on a fresh start
                elif entry.get("epoch") == len(valid_log) + 1:
                    valid_log.append(entry)
                    
            # Truncate to match the checkpoint exactly
            training_log = valid_log[:start_epoch]
            
            if is_master: print(f"[Train] Loaded and cleaned training log ({len(training_log)} epochs) to match checkpoint.")
        except Exception as e:
            print(f"[-] Failed to load training log: {e}")

    if getattr(args, 'multi_gpu', False) and torch.cuda.device_count() > 1:
        dist.barrier()
        if not is_master and os.path.exists(latest_ckpt_path) and not getattr(args, 'fresh', False):
            # load it on workers too!
            ckpt = torch.load(latest_ckpt_path, map_location=device)
            get_raw_module(base_G).load_state_dict(ckpt["G"])
            get_raw_module(base_D).load_state_dict(ckpt["D"])
            opt_G.load_state_dict(ckpt["opt_G"])
            opt_D.load_state_dict(ckpt["opt_D"])
            # Override learning rates with command line arguments on resume
            for g in opt_G.param_groups: g['lr'] = args.lr_g
            for g in opt_D.param_groups: g['lr'] = args.lr_d
            start_epoch = ckpt["epoch"] + 1

    # ---- Training Loop ----
    for epoch in range(start_epoch, args.epochs):
        if sampler is not None: sampler.set_epoch(epoch)
        G.train()
        D.train()

        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        epoch_r1 = 0.0
        num_batches = 0
        t0 = time.time()

        for batch_idx, (real_images, _) in enumerate(dataloader):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # ========================
            # Train Discriminator (VRAM-efficient & numerically stable mixed precision)
            # ========================
            D.zero_grad()

            # Main D loss: run in mixed precision (autocast) to drastically reduce VRAM activations
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                real_aug = DiffAugment(real_images, policy=args.augment) if args.augment else real_images
                real_score = D(real_aug)

                # Fake images (keep fake forward pass inside autocast)
                z = torch.randn(batch_size, args.noise_dim, device=device)
                with torch.no_grad():
                    fake_images = G(z)
                fake_aug = DiffAugment(fake_images, policy=args.augment) if args.augment else fake_images
                fake_score = D(fake_aug.detach())

                # Main hinge loss
                d_loss = hinge_loss_dis(real_score, fake_score)
            
            # R1 penalty (every 4 steps): run in float32 (outside autocast)
            r1_loss = torch.tensor(0.0, device=device)
            if batch_idx % 4 == 0:
                with torch.amp.autocast('cuda', enabled=False):
                    # VRAM Optimization: R1 double-backward in float32 is extremely memory intensive.
                    # Compute it on a sub-batch (max 32 images) to prevent OOM. Regularization still works perfectly.
                    r1_bs = min(batch_size, 32)
                    real_images_f32 = real_images[:r1_bs].float().detach().requires_grad_(True)
                    
                    # Temporarily set D to eval mode. 
                    # This prevents spectral_norm from running power iterations during the R1 forward pass.
                    # Backpropagating double-derivatives through power iteration loops causes massive NaN explosions!
                    D.eval()
                    real_score_r1 = D(real_images_f32)
                    D.train()
                    
                    r1_loss = r1_penalty(real_images_f32, real_score_r1)
                    
                    scaled_r1_loss = r1_loss * args.r1_weight * scaler_D.get_scale()
                    scaled_r1_loss.backward()

            # Scale and backward the main loss
            scaler_D.scale(d_loss).backward()
            
            # Manually synchronize Discriminator gradients across all GPUs safely
            if use_ddp:
                for param in D.parameters():
                    if param.grad is not None:
                        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                        param.grad /= dist.get_world_size()

            # Optimizer step & update scaler
            prev_scale_d = scaler_D.get_scale()
            scaler_D.step(opt_D)
            scaler_D.update()
            if scaler_D.get_scale() < prev_scale_d:
                d_scaler_skips += 1

            # ========================
            # Train Generator (multiple G steps per D step for balance)
            # ========================
            for _g_step in range(args.g_steps):
                G.zero_grad()

                z = torch.randn(batch_size, args.noise_dim, device=device)

                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    fake_images = G(z)
                    fake_aug = DiffAugment(fake_images, policy=args.augment) if args.augment else fake_images
                    g_fake_score = D(fake_aug)
                    g_loss = hinge_loss_gen(g_fake_score)

                prev_scale = scaler_G.get_scale()
                scaler_G.scale(g_loss).backward()
                scaler_G.step(opt_G)
                scaler_G.update()
                if scaler_G.get_scale() < prev_scale:
                    g_scaler_skips += 1

            # Sanitize BatchNorm buffers (prevents AMP overflow black image bug)
            sanitize_bn_buffers(get_raw_module(base_G))

            # Update EMA (master only — used only for sample generation)
            if is_master:
                ema.update(get_raw_module(base_G))

            # Accumulate stats
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            epoch_r1 += r1_loss.item()
            num_batches += 1

            # Progress bar (with D score diagnostics)
            if batch_idx % 20 == 0:
                elapsed = time.time() - t0
                d_real = real_score.mean().item()
                d_fake = fake_score.mean().item()
                print(
                    f"\r  Epoch [{epoch+1}/{args.epochs}] "
                    f"Batch [{batch_idx+1}/{len(dataloader)}] "
                    f"D: {d_loss.item():.4f}  G: {g_loss.item():.4f}  "
                    f"R1: {r1_loss.item():.4f}  "
                    f"D(real): {d_real:+.3f}  D(fake): {d_fake:+.3f}  "
                    f"Time: {elapsed:.1f}s",
                    end="", flush=True,
                )

        # Epoch stats
        epoch_time = time.time() - t0
        avg_d = epoch_d_loss / max(num_batches, 1)
        avg_g = epoch_g_loss / max(num_batches, 1)
        avg_r1 = epoch_r1 / max(num_batches, 1)

        ram_gb = psutil.virtual_memory().used / (1024 ** 3)
        vram_gb = torch.cuda.memory_reserved(device) / (1024 ** 3) if torch.cuda.is_available() else 0.0
        num_gpus = torch.cuda.device_count()

        if is_master:
            print(
                f"\n  ✓ Epoch {epoch+1}/{args.epochs} — "
                f"D_loss: {avg_d:.4f}  G_loss: {avg_g:.4f}  "
                f"R1: {avg_r1:.4f}  Time: {epoch_time:.1f}s  "
                f"RAM: {ram_gb:.1f}GB  VRAM: {vram_gb:.1f}GB/GPU ×{num_gpus}  "
                f"G_scaler_skips: {g_scaler_skips}  D_scaler_skips: {d_scaler_skips}"
            )

        # Log
        training_log.append({
            "epoch": epoch + 1,
            "d_loss": avg_d,
            "g_loss": avg_g,
            "r1": avg_r1,
            "time": epoch_time,
            "ram_gb": ram_gb,
            "vram_gb": vram_gb
        })

        # ============================================================
        # SYNC POINT: Check stop signal BEFORE any master-only I/O.
        # This prevents rank 1 from timing out while rank 0 uploads
        # checkpoints to Google Drive (which can take 10+ minutes).
        # ============================================================
        stop_flag = torch.tensor([0], device=device)
        if is_master and drive_sync and drive_sync.check_and_delete_file(f"stop_{args.session}.txt"):
            stop_flag[0] = 1
        if use_ddp:
            dist.broadcast(stop_flag, 0)
            dist.barrier()  # Ensure all ranks are synced before master does I/O

        # ---- Master-only I/O (rank 1 will wait at next epoch's DDP forward) ----

        # ---- Generate Samples ----
        if is_master and ((epoch + 1) % args.sample_every == 0 or epoch == 0):
            base_G.eval()
            ema.apply_shadow(get_raw_module(base_G))

            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    samples = base_G(fixed_noise)
                
                # --- Anti-Black-Image Diagnostic ---
                if torch.isnan(samples).any() or torch.isinf(samples).any():
                    print(f"  [BUG] ⚠️ samples contain NaN/Inf!")
                if samples.max() == samples.min():
                    print(f"  [BUG] ⚠️ samples are completely uniform (min={samples.min().item():.3f}, max={samples.max().item():.3f})! Checking EMA...")
                    has_nan = any(torch.isnan(v).any() for v in ema.shadow.values())
                    print(f"  [BUG] EMA shadow has NaN: {has_nan}")

                samples = samples.float().clamp(-1, 1)
                samples = (samples + 1) / 2
                grid = make_grid(samples, nrow=4)

            sample_img_path = os.path.join(args.sample_dir, f"epoch_{epoch+1:04d}.png")
            save_image(grid, sample_img_path)
            print(f"  📸 Saved samples → {sample_img_path}")
            drive_sync.upload_file(sample_img_path, f"epoch_{epoch+1:04d}_{args.session}.png")
            drive_sync.upload_file(sample_img_path, f"latest_sample_{args.session}.png")

            ema.restore(get_raw_module(base_G))
            base_G.train()

        # ---- Save Checkpoint (Master only) ----
        if is_master:
            ckpt = {
                "epoch": epoch,
                "G": get_raw_module(base_G).state_dict(),
                "D": get_raw_module(base_D).state_dict(),
                "opt_G": opt_G.state_dict(),
                "opt_D": opt_D.state_dict(),
                "ema": ema.shadow,
                "args": vars(args),
            }
            # Save locally every epoch (instant, ~1s)
            torch.save(ckpt, latest_ckpt_path)

            # Upload to Drive only every save_every epochs (avoids 10-min upload stalls)
            if (epoch + 1) % args.save_every == 0:
                ckpt_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch+1:04d}.pt")
                torch.save(ckpt, ckpt_path)
                drive_sync.upload_file(latest_ckpt_path, latest_ckpt_name)
                print(f"  💾 Checkpoint uploaded to Drive (epoch {epoch+1})")

            # Log JSON is tiny — upload every epoch
            with open(log_path, "w") as f:
                json.dump(training_log, f, indent=2)
            drive_sync.upload_file(log_path, log_name)

        # Break AFTER I/O so checkpoint is saved before exit
        if stop_flag.item() == 1:
            if is_master: print(f"[Train] Stop signal received for session '{args.session}'. Exiting cleanly.")
            break

    if is_master:
        print(f"\n{'='*60}")
        print(f"  🎉 Training complete!")
        print(f"  Checkpoints: {args.checkpoint_dir}")
        print(f"  Samples: {args.sample_dir}")
        
        # Save finalized Generator weights (EMA weights) directly for easy inference
        final_gen_path = os.path.join(args.checkpoint_dir, f"generator_final_{args.session}.pt")
        ema.apply_shadow(base_G)
        torch.save(get_raw_module(base_G).state_dict(), final_gen_path)
        ema.restore(base_G)
        print(f"  💾 Finalized Generator (EMA) weights saved to {final_gen_path}")
        print(f"{'='*60}\n")

    # Cleanup DDP
    if use_ddp:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Train Dog-GAN")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to Stanford Dogs dataset")
    parser.add_argument("--download", action="store_true", help="Download dataset if not present")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--sample_dir", type=str, default="./samples")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--noise_dim", type=int, default=128)
    parser.add_argument("--lr_g", type=float, default=1e-4, help="Generator learning rate")
    parser.add_argument("--lr_d", type=float, default=4e-4, help="Discriminator learning rate (TTUR)")
    parser.add_argument("--r1_weight", type=float, default=5.0, help="R1 gradient penalty weight")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--sample_every", type=int, default=5, help="Generate samples every N epochs")
    parser.add_argument("--save_every", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--session", type=str, default="default", help="Session name for checkpoint naming")
    parser.add_argument("--multi_gpu", action="store_true", help="Enable DDP across multiple GPUs")
    parser.add_argument("--model_v2", action="store_true", help="Use V2 high-capacity architecture (~24M params)")
    parser.add_argument("--augment", type=str, default="color,translation,cutout", help="DiffAugment policy (e.g. 'color,translation,cutout')")
    parser.add_argument("--fresh", action="store_true", help="Start from scratch, ignore any existing checkpoint")
    parser.add_argument("--g_steps", type=int, default=2, help="Generator updates per Discriminator update")

    args = parser.parse_args()
    
    if args.multi_gpu and torch.cuda.device_count() > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '23456'
        mp.spawn(train_worker, nprocs=torch.cuda.device_count(), args=(args,))
    else:
        train_worker(0, args)

if __name__ == "__main__":
    main()
