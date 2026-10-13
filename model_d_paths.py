from pathlib import Path

# Existing synthetic dataset.
DATA_ROOT = Path(
    "/data/VAE/maestro_fresh/maestro_tempo_pilot"
)

# Existing pretrained VAE.
VAE_CHECKPOINT = Path(
    "/data/VAE/HarmonAI/sample-generator-main-1/"
    "vae_checkpoints/epoch=7499-step=50000.ckpt"
)

# Existing Model D checkpoints.
CHECKPOINTS = {
    "epoch500": Path(
        "/data/VAE/HarmonAI/sample-generator-main-1/"
        "model_d_synthetic_500epoch_checkpoints/last.ckpt"
    ),
}

# Separate output folder for checking the organized repository.
OUTPUT_ROOT = Path(
    "/data/VAE/maestro_fresh/model_d_30second_generation_clean"
)

# Default location for any future training runs.
TRAIN_CHECKPOINT_DIR = Path(
    "/data/VAE/maestro_fresh/model_d_checkpoints"
)

WANDB_DIR = Path(
    "/data/VAE/maestro_fresh/wandb"
)