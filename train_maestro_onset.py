#!/usr/bin/env python3
# Model D: fixed pretrained VAE + 1D diffusion UNet + MIDI note-onset timing.
# Requires dataset/onset_grid_features.py and dataset/steady_onset_dataset.py.
from dataset.onset_grid_features import make_scalar_onset_condition
from audio_diffusion.vae1d import AudioVAE1D
from prefigure.prefigure import get_all_args, push_wandb_config
from contextlib import contextmanager
from copy import deepcopy
import math
from pathlib import Path

import sys
import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils import data
from tqdm import trange
import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from einops import rearrange
import torchaudio
import wandb

from dataset.synthetic_maestro_tempo_dataset import SyntheticMaestroTempoDataset

from audio_diffusion.models import DiffusionAttnUnet1D
from audio_diffusion.utils import ema_update
from viz.viz import audio_spectrogram_image
from model_d_paths import (
    DATA_ROOT,
    VAE_CHECKPOINT,
    TRAIN_CHECKPOINT_DIR,
    WANDB_DIR,
)

# Define the noise schedule and sampling loop
def get_alphas_sigmas(t):
    """Returns the scaling factors for the clean image (alpha) and for the
    noise (sigma), given a timestep."""
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)

def get_crash_schedule(t):
    sigma = torch.sin(t * math.pi / 2) ** 2
    alpha = (1 - sigma ** 2) ** 0.5
    return alpha_sigma_to_t(alpha, sigma)

def alpha_sigma_to_t(alpha, sigma):
    """Returns a timestep, given the scaling factors for the clean image and for
    the noise."""
    return torch.atan2(sigma, alpha) / math.pi * 2

@torch.no_grad()
def sample(
    model,
    x,
    steps,
    eta,
    cond=None,
    guidance_scale=1.0,
):
    """
    Draw samples from a model given starting noise.

    cond shape:
        [B, condition_channels, latent_time]
    """
    ts = x.new_ones([x.shape[0]])

    t = torch.linspace(
        1,
        0,
        steps + 1,
        device=x.device,
    )[:-1]

    t = get_crash_schedule(t)

    alphas, sigmas = get_alphas_sigmas(t)

    for i in trange(steps):
        current_t = ts * t[i]

        with torch.cuda.amp.autocast():
            if cond is not None and guidance_scale != 1.0:
                v_uncond = model(
                    x,
                    current_t,
                    cond=torch.zeros_like(cond),
                ).float()

                v_cond = model(
                    x,
                    current_t,
                    cond=cond,
                ).float()

                v = v_uncond + guidance_scale * (v_cond - v_uncond)

            else:
                v = model(
                    x,
                    current_t,
                    cond=cond,
                ).float()

        pred = x * alphas[i] - v * sigmas[i]
        eps = x * sigmas[i] + v * alphas[i]

        if i < steps - 1:
            ddim_sigma = (
                eta
                * (sigmas[i + 1] ** 2 / sigmas[i] ** 2).sqrt()
                * (1 - alphas[i] ** 2 / alphas[i + 1] ** 2).sqrt()
            )

            adjusted_sigma = (
                sigmas[i + 1] ** 2
                - ddim_sigma ** 2
            ).sqrt()

            x = (
                pred * alphas[i + 1]
                + eps * adjusted_sigma
            )

            if eta:
                x += torch.randn_like(x) * ddim_sigma

    return pred



class DiffusionUncond(pl.LightningModule):
    def __init__(self, global_args):
        super().__init__()

        self.global_args = global_args
        self.rng = torch.quasirandom.SobolEngine(
        1,
        scramble=True,
        seed=global_args.seed
        )
        self.ema_decay = global_args.ema_decay

    # -----------------------------
    # Load trained VAE
    # -----------------------------
        self.vae = AudioVAE1D(
        in_channels=2,
        latent_channels=global_args.latent_channels,
        base_channels=global_args.vae_base_channels,
       )

        ckpt = torch.load(global_args.vae_ckpt_path, map_location="cpu")
        state_dict = ckpt["state_dict"]

        vae_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("vae."):
                vae_state_dict[key.replace("vae.", "")] = value

        if len(vae_state_dict) == 0:
            raise RuntimeError(
            "No VAE weights found in checkpoint. Expected keys starting with 'vae.'. "
            "Make sure vae_ckpt_path points to a checkpoint created by train_vae.py."
            )

        self.vae.load_state_dict(vae_state_dict)
        self.vae.eval()

        for p in self.vae.parameters():
            p.requires_grad = False

    # -----------------------------
    # Latent diffusion model
    # -----------------------------
        self.diffusion = DiffusionAttnUnet1D(
        global_args,
        io_channels=global_args.latent_channels,
        depth=6,
        n_attn_layers=3,
       )

    # -----------------------------
    # EMA copy of diffusion model
    # -----------------------------
        self.diffusion_ema = deepcopy(self.diffusion)

        for p in self.diffusion_ema.parameters():
            p.requires_grad = False

        self.diffusion_ema.eval()
        
    def configure_optimizers(self):
        return optim.Adam([*self.diffusion.parameters()], lr=4e-5)
  
    def training_step(self, batch, batch_idx):
        # The dataset returns: stereo WAV, normalized BPM, padded MIDI onset times, path.
        if len(batch) != 4:
            raise RuntimeError("Expected (audio, tempo, onsets, filename)")

        reals, tempo_vec, midi_onsets, _filenames = batch
        tempo_vec = tempo_vec.to(self.device)
        midi_onsets = midi_onsets.to(self.device)

        # Lightning .train() may toggle submodules; always keep pretrained VAE fixed.
        self.vae.eval()
        with torch.no_grad():
            mu, _logvar = self.vae.encode(reals)
            latents = mu

        t = self.rng.draw(latents.shape[0])[:, 0].to(self.device)
        t = get_crash_schedule(t)
        alphas, sigmas = get_alphas_sigmas(t)
        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]

        noise = torch.randn_like(latents)
        noised_latents = latents * alphas + noise * sigmas
        targets = noise * alphas - latents * sigmas

        cond = make_scalar_onset_condition(
            tempo_vec,
            midi_onsets,
            latent_length=noised_latents.shape[-1],
            sample_rate=self.global_args.sample_rate,
            downsample_factor=16,
            onset_sigma=0.03,
        )

        expected = (latents.shape[0], 2, latents.shape[-1])
        if cond.shape != expected:
            raise RuntimeError(
                "Bad Model D condition shape: {} != {}".format(cond.shape, expected)
            )

        if self.global_step == 0 and batch_idx == 0:
            print("Model D audio:", tuple(reals.shape))
            print("Model D latent:", tuple(latents.shape))
            print("Model D condition:", tuple(cond.shape))
            print("Model D normalized BPM:", float(tempo_vec[0, 0]))
            print("Model D example note count:", int((midi_onsets[0] >= 0).sum()))

        with torch.cuda.amp.autocast():
            v = self.diffusion(noised_latents, t, cond=cond)
            loss = F.mse_loss(v, targets)

        if self.global_step == 0 and batch_idx == 0:
            print("DIAGNOSTIC timestep:",
                  t.detach().float().flatten().cpu().tolist())
            print("DIAGNOSTIC alpha:",
                  alphas.detach().float().flatten().cpu().tolist())
            print("DIAGNOSTIC sigma:",
                  sigmas.detach().float().flatten().cpu().tolist())
            print("DIAGNOSTIC audio RMS:",
                  reals.float().square().mean().sqrt().item())
            print("DIAGNOSTIC latent mean:",
                  latents.float().mean().item())
            print("DIAGNOSTIC latent std:",
                  latents.float().std().item())
            print("DIAGNOSTIC target power:",
                  targets.float().square().mean().item())
            print("DIAGNOSTIC prediction power:",
                  v.float().square().mean().item())
            print("DIAGNOSTIC initial MSE:",
                  loss.float().item())

        self.log_dict(
            {"train/loss": loss.detach(), "train/mse_loss": loss.detach()},
            prog_bar=True, on_step=True,
        )
        return loss


    def validation_step(self, batch, batch_idx):
        reals, tempo_vec, midi_onsets, _ = batch

        self.vae.eval()

        with torch.no_grad():
            latents, _ = self.vae.encode(reals)

        b = latents.shape[0]

        # Fixed validation timesteps for consistent comparisons.
        raw_t = float((batch_idx % 19) + 1) / 20.0
        t = torch.full(
            (b,), raw_t, device=self.device
        )
        t = get_crash_schedule(t)

        alpha, sigma = get_alphas_sigmas(t)
        alpha = alpha[:, None, None]
        sigma = sigma[:, None, None]

        generator = torch.Generator(device=self.device)
        generator.manual_seed(12345 + batch_idx)

        noise = torch.randn(
            latents.shape,
            device=self.device,
            dtype=latents.dtype,
            generator=generator,
        )

        noised = latents * alpha + noise * sigma
        target = noise * alpha - latents * sigma

        cond = make_scalar_onset_condition(
            tempo_vec.to(self.device),
            midi_onsets.to(self.device),
            latent_length=latents.shape[-1],
            sample_rate=self.global_args.sample_rate,
            downsample_factor=16,
            onset_sigma=0.03,
        )

        with torch.cuda.amp.autocast():
            prediction = self.diffusion(
                noised, t, cond=cond
            )

        loss = F.mse_loss(
            prediction.float(), target.float()
        )

        self.log(
            "val/mse",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=b,
        )
        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        decay = 0.95 if self.current_epoch < 25 else self.ema_decay
        ema_update(self.diffusion, self.diffusion_ema, decay)

class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}', file=sys.stderr)



def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)

    parser.add_argument(
        "--data-dir",
        default=str(DATA_ROOT),
    )
    parser.add_argument(
        "--project", default="harmonai-model-d"
    )
    parser.add_argument(
        "--run-name", default="model-d-synthetic-5k"
    )
    parser.add_argument(
        "--ckpt-dir",
        default=str(TRAIN_CHECKPOINT_DIR),
    )
    
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--wandb-id", default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--vae-ckpt",
        default=str(VAE_CHECKPOINT),
    )
    parser.add_argument(
        "--vae-ckpt",
        default=str(VAE_CHECKPOINT),
    )
    cli = parser.parse_args()

    if cli.epochs < 1 or cli.batch_size < 1 or cli.devices < 1:
        parser.error("epochs, batch-size and devices must be positive")

    if cli.max_steps != -1 and cli.max_steps < 1:
        parser.error("max-steps must be positive or -1")

    if cli.save_every < 1 or cli.num_workers < 0:
        parser.error("Invalid save-every or num-workers")

    if cli.resume and not Path(cli.resume).is_file():
        raise FileNotFoundError(cli.resume)

    ckpt_dir = Path(cli.ckpt_dir)
    if not cli.smoke:
        if not cli.resume and ckpt_dir.exists():
            if list(ckpt_dir.glob("*.ckpt")):
                raise RuntimeError(
                    "Existing checkpoints found. "
                    "Use --resume or another --ckpt-dir."
                )
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Load Prefigure defaults without passing our CLI options.
    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        args = get_all_args()
    finally:
        sys.argv = original_argv

    args.latent_dim = 2
    args.latent_channels = 8
    args.vae_base_channels = 64
    args.vae_ckpt_path = str(
        Path(cli.vae_ckpt).expanduser().resolve()
    )
    args.sample_rate = 8000
    args.sample_size = 65536
    args.min_bpm = 60
    args.max_bpm = 180

    args.seed = cli.seed
    args.name = cli.run_name
    args.save_path = str(ckpt_dir)
    args.ckpt_path = ""
    args.batch_size = cli.batch_size
    args.accum_batches = 1
    args.cond_drop_prob = 0.0

    pl.seed_everything(cli.seed, workers=True)

    train_set = SyntheticMaestroTempoDataset(
        base_dir=cli.data_dir,
        split="train",
        min_bpm=60,
        max_bpm=180,
    )
    val_set = SyntheticMaestroTempoDataset(
        base_dir=cli.data_dir,
        split="validation",
        min_bpm=60,
        max_bpm=180,
    )

    assert len(train_set) == 108
    assert len(val_set) == 36
    assert train_set.max_onsets == val_set.max_onsets

    train_dl = data.DataLoader(
        train_set,
        batch_size=cli.batch_size,
        shuffle=True,
        num_workers=0 if cli.smoke else cli.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_dl = data.DataLoader(
        val_set,
        batch_size=cli.batch_size,
        shuffle=False,
        num_workers=0 if cli.smoke else cli.num_workers,
        pin_memory=True,
    )

    model = DiffusionUncond(args)

    if cli.smoke:
        logger = False
        callbacks = [ExceptionCallback()]
    else:
        wandb_dir = WANDB_DIR
        wandb_dir.mkdir(parents=True, exist_ok=True)

        logger_args = dict(
            project=cli.project,
            name=cli.run_name,
            save_dir=str(wandb_dir),
            log_model=False,
            offline=cli.offline,
        )

        if cli.wandb_id:
            logger_args["id"] = cli.wandb_id
            logger_args["resume"] = "allow"

        logger = pl.loggers.WandbLogger(**logger_args)

        logger.log_hyperparams({
            "model": "D",
            "dataset": cli.data_dir,
            "training_clips": len(train_set),
            "validation_clips": len(val_set),
            "epochs": cli.epochs,
            "max_steps": cli.max_steps,
            "batch_size": cli.batch_size,
            "learning_rate": 4e-5,
            "bpm_range": "60-180",
            "conditioning": "scalar_bpm_and_midi_onsets",
            "synthetic_audio": True,
            "vae_frozen": True,
            "resume_checkpoint": cli.resume or "none",
        })

        checkpoint = pl.callbacks.ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="epoch{epoch:03d}-step{step}",
            auto_insert_metric_name=False,
            every_n_train_steps=cli.save_every,
            save_top_k=-1,
            save_last=True,
        )
        callbacks = [checkpoint, ExceptionCallback()]

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=cli.devices,
        precision=16,
        max_epochs=cli.epochs,
        max_steps=1 if cli.smoke else cli.max_steps,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        fast_dev_run=1 if cli.smoke else False,
        enable_checkpointing=not cli.smoke,
    )

    print("\nMODEL D CONFIGURATION")
    print("Train / validation:", len(train_set), len(val_set))
    print("Epoch limit:", cli.epochs)
    print("Step limit:", 1 if cli.smoke else cli.max_steps)
    print("Checkpoint directory:", ckpt_dir)
    print("Resume:", cli.resume or "fresh initialization")
    print("W&B:", "off (smoke)" if cli.smoke else cli.project)

    trainer.fit(
        model,
        train_dl,
        val_dl,
        ckpt_path=cli.resume if not cli.smoke else None,
    )

    print("MODEL D TRAINING FINISHED")


if __name__ == "__main__":
    main()
