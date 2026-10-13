
#!/usr/bin/env python3
"""
MODEL D — 30-SECOND AUDIO GENERATION

Generates 30-second piano audio using existing Model D checkpoints.

Configuration:
    - Multiple checkpoints
    - 9 BPM values (60-180)
    - 4 rhythmic patterns
    - 30-second duration
    - Stereo, 8000 Hz
    - Reproducible random seeds

IMPORTANT:
Model D was trained on 8.192-second audio.
This script generates short segments and joins them using
crossfading to create approximately continuous 30-second audio.

It does not provide native 30-second diffusion generation
or guarantee musical continuity between segments.
"""

import json
import math
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf
import torch

from dataset.onset_grid_features import make_scalar_onset_condition
from generate_tempo import load_diffusion, load_vae
from train_maestro_onset import sample


# ============================================================
# 1. CONFIGURATION — EDIT THESE VARIABLES
# ============================================================

from model_d_paths import (
    DATA_ROOT,
    OUTPUT_ROOT,
    VAE_CHECKPOINT,
    CHECKPOINTS,
)

DURATION_SECONDS = 30.0
SEED = 2026
SAMPLING_STEPS = 100
ETA = 0.0
GUIDANCE_SCALE = 1.0
OVERWRITE = False

SAMPLE_RATE = 8000
CHUNK_SAMPLES = 65536
LATENT_LENGTH = 4096
LATENT_CHANNELS = 8
MAX_MIDI_ONSETS = 47

# ------------------------------------------------------------
# TEMPO CONFIGURATION
# ------------------------------------------------------------

TEMPOS_BPM = (
    60,
    75,
    90,
    105,
    120,
    135,
    150,
    165,
    180,
)


# ------------------------------------------------------------
# RHYTHMIC PATTERNS
# ------------------------------------------------------------

PATTERNS = (
    "steady",
    "rests",
    "mixed",
    "eighth",
)


# ------------------------------------------------------------
# DATASET REFERENCE
# ------------------------------------------------------------

DATA_SPLIT = "validation"

PERFORMANCE = "performance03"


# ------------------------------------------------------------
# GENERATION SETTINGS
# ------------------------------------------------------------

DURATION_SECONDS = 30.0

SEED = 2026

SAMPLING_STEPS = 100

ETA = 0.0

GUIDANCE_SCALE = 1.0

OVERWRITE = False

OUTPUT_GAIN = 1.0


# ============================================================
# 2. MODEL CONSTANTS
# ============================================================

SAMPLE_RATE = 8000

CHUNK_SAMPLES = 65536

LATENT_LENGTH = 4096

LATENT_CHANNELS = 8

MAX_MIDI_ONSETS = 47

CHUNK_DURATION = CHUNK_SAMPLES / SAMPLE_RATE

# 8 seconds is an integer number of beats for every trained BPM.
CHUNK_HOP_SECONDS = 8.0

HOP_SAMPLES = int(
    round(CHUNK_HOP_SECONDS * SAMPLE_RATE)
)

OVERLAP_SAMPLES = CHUNK_SAMPLES - HOP_SAMPLES


# ============================================================
# 3. READ MIDI NOTE ONSETS
# ============================================================

def read_midi_onsets(midi_path, expected_bpm):

    midi = pretty_midi.PrettyMIDI(
        str(midi_path)
    )

    _, tempo_values = midi.get_tempo_changes()

    if not len(tempo_values):
        raise ValueError(
            "MIDI contains no tempo information"
        )

    for value in tempo_values:
        if abs(float(value) - expected_bpm) > 0.1:
            raise ValueError(
                "MIDI tempo does not match expected BPM: "
                + str(midi_path)
            )

    events = sorted(set(

        round(float(note.start), 6)

        for instrument in midi.instruments

        for note in instrument.notes

        if 0 <= note.start < CHUNK_DURATION

    ))

    if not 0 < len(events) <= MAX_MIDI_ONSETS:

        raise ValueError(
            "Invalid MIDI onset count: "
            + str(len(events))
        )

    return events


# ============================================================
# 4. CROSSFADE AUDIO SEGMENTS
# ============================================================

def overlap_append(previous, current):

    if previous is None:
        return current.copy()

    if previous.ndim != 2:
        raise ValueError(
            "Expected stereo audio"
        )

    if current.shape != (CHUNK_SAMPLES, 2):
        raise ValueError(
            "Unexpected chunk shape"
        )

    alpha = np.linspace(
        0.0,
        1.0,
        OVERLAP_SAMPLES,
        dtype=np.float32
    )[:, None]

    previous_overlap = (
        previous[-OVERLAP_SAMPLES:]
    )

    current_overlap = (
        current[:OVERLAP_SAMPLES]
    )

    mixed = (
        previous_overlap * (1.0 - alpha)
        +
        current_overlap * alpha
    )

    combined = np.concatenate(

        (
            previous[:-OVERLAP_SAMPLES],
            mixed,
            current[OVERLAP_SAMPLES:]
        ),

        axis=0

    )

    return combined


# ============================================================
# 5. GENERATE ONE 30-SECOND AUDIO FILE
# ============================================================

def generate_30_seconds(

    diffusion,
    vae,
    bpm,
    pattern,
    checkpoint_name,
    midi_path,
    reference_path,
    output_path

):

    print(
        "\nGenerating:",
        checkpoint_name,
        pattern,
        bpm,
        "BPM"
    )

    onset_times = read_midi_onsets(
        midi_path,
        bpm
    )

    required_samples = int(
        round(DURATION_SECONDS * SAMPLE_RATE)
    )

    # Calculate number of 8.192-second chunks.
    number_of_chunks = max(

        1,

        1 + math.ceil(

            max(
                0,
                required_samples - CHUNK_SAMPLES
            ) / HOP_SAMPLES

        )

    )

    print(
        "Number of chunks:",
        number_of_chunks
    )

    print(
        "MIDI onset count:",
        len(onset_times)
    )


    # --------------------------------------------------------
    # BPM NORMALIZATION
    # --------------------------------------------------------

    # Model D training used 60-180 BPM mapped to [-1, 1].

    normalized_bpm = (

        2.0 * (bpm - 60.0) / 120.0 - 1.0

    )

    scalar_bpm = torch.tensor(

        [[normalized_bpm]],

        dtype=torch.float32,

        device="cuda"

    )


    # --------------------------------------------------------
    # MIDI ONSET CONDITIONING
    # --------------------------------------------------------

    onset_array = torch.full(

        (1, MAX_MIDI_ONSETS),

        -1.0,

        dtype=torch.float32,

        device="cuda"

    )

    onset_array[0, :len(onset_times)] = torch.tensor(

        onset_times,

        dtype=torch.float32,

        device="cuda"

    )


    condition = make_scalar_onset_condition(

        scalar_bpm,

        onset_array,

        latent_length=LATENT_LENGTH,

        sample_rate=SAMPLE_RATE,

        downsample_factor=16,

        onset_sigma=0.03

    )

    if condition.shape != (1, 2, LATENT_LENGTH):

        raise RuntimeError(
            "Incorrect conditioning dimensions"
        )


    # --------------------------------------------------------
    # GENERATE AUDIO CHUNKS
    # --------------------------------------------------------

    combined_audio = None

    chunk_information = []

    for chunk_index in range(number_of_chunks):

        chunk_seed = SEED + chunk_index

        torch.manual_seed(chunk_seed)

        torch.cuda.manual_seed_all(chunk_seed)


        # ----------------------------------------------------
        # INITIAL LATENT NOISE
        # ----------------------------------------------------

        noise = torch.randn(

            (
                1,
                LATENT_CHANNELS,
                LATENT_LENGTH
            ),

            device="cuda"

        )


        # ----------------------------------------------------
        # DIFFUSION SAMPLING
        # ----------------------------------------------------

        with torch.no_grad():

            generated_latents = sample(

                diffusion,

                noise,

                steps=SAMPLING_STEPS,

                eta=ETA,

                cond=condition,

                guidance_scale=GUIDANCE_SCALE

            )


            # ------------------------------------------------
            # VAE DECODING
            # ------------------------------------------------

            decoded_audio = vae.decode(
                generated_latents
            )


        expected_shape = (
            1,
            2,
            CHUNK_SAMPLES
        )

        if tuple(decoded_audio.shape) != expected_shape:

            raise RuntimeError(
                "Unexpected decoded audio shape: "
                + str(tuple(decoded_audio.shape))
            )

        if not torch.isfinite(decoded_audio).all():

            raise RuntimeError(
                "Generated audio contains invalid values"
            )


        # ----------------------------------------------------
        # CONVERT TO NUMPY
        # ----------------------------------------------------

        chunk_audio = (

            decoded_audio[0]

            .float()

            .detach()

            .cpu()

            .numpy()

            .T

        )


        # ----------------------------------------------------
        # JOIN WITH CROSSFADE
        # ----------------------------------------------------

        combined_audio = overlap_append(

            combined_audio,

            chunk_audio

        )


        chunk_information.append({

            "chunk": chunk_index,

            "seed": chunk_seed,

            "start_seconds": round(

                chunk_index * CHUNK_HOP_SECONDS,

                6

            )

        })


        print(
            "Completed chunk",
            chunk_index + 1,
            "/",
            number_of_chunks
        )


        del generated_latents
        del decoded_audio


    # --------------------------------------------------------
    # TRIM TO 30 SECONDS
    # --------------------------------------------------------

    final_audio = (

        combined_audio[:required_samples]

        * OUTPUT_GAIN

    )

    if len(final_audio) != required_samples:

        raise RuntimeError(
            "Final audio duration is incorrect"
        )


    # --------------------------------------------------------
    # CHECK AMPLITUDE
    # --------------------------------------------------------

    peak_amplitude = float(

        np.max(np.abs(final_audio))

    )

    clipped_fraction = float(

        np.mean(np.abs(final_audio) >= 1.0)

    )


    # --------------------------------------------------------
    # SAVE WAV
    # --------------------------------------------------------

    final_audio = np.clip(

        final_audio,

        -1.0,

        1.0

    )

    sf.write(

        str(output_path),

        final_audio,

        SAMPLE_RATE,

        subtype="PCM_16"

    )


    # --------------------------------------------------------
    # SAVE METADATA
    # --------------------------------------------------------

    metadata = {

        "model": "Model D",

        "checkpoint_name": checkpoint_name,

        "checkpoint": str(
            CHECKPOINTS[checkpoint_name]
        ),

        "vae_checkpoint": str(
            VAE_CHECKPOINT
        ),

        "tempo_bpm": bpm,

        "normalized_scalar_bpm": normalized_bpm,

        "pattern": pattern,

        "duration_seconds": DURATION_SECONDS,

        "sample_rate": SAMPLE_RATE,

        "sampling_steps": SAMPLING_STEPS,

        "seed": SEED,

        "midi_path": str(midi_path),

        "reference_wav": str(reference_path),

        "midi_onsets_per_chunk_seconds": onset_times,

        "number_of_chunks": number_of_chunks,

        "chunk_duration_seconds": CHUNK_DURATION,

        "chunk_hop_seconds": CHUNK_HOP_SECONDS,

        "crossfade_seconds": (
            OVERLAP_SAMPLES / SAMPLE_RATE
        ),

        "chunks": chunk_information,

        "peak_before_clipping": peak_amplitude,

        "fraction_outside_pcm_range": clipped_fraction,

        "output_path": str(output_path),

        "note": (
            "Audio assembled from independently generated "
            "short Model D samples with crossfades."
        )

    }


    json_path = output_path.with_suffix(
        ".json"
    )

    json_path.write_text(

        json.dumps(
            metadata,
            indent=2
        )

    )


    print(
        "SAVED:",
        output_path
    )


# ============================================================
# 6. MAIN GENERATION PIPELINE
# ============================================================

def main():

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA GPU is required"
        )

    if not CHECKPOINTS:

        raise ValueError(
            "No checkpoints configured"
        )

    if DURATION_SECONDS <= 0 or SAMPLING_STEPS < 1:

        raise ValueError(
            "Invalid generation settings"
        )


    # --------------------------------------------------------
    # VERIFY CHECKPOINT FILES
    # --------------------------------------------------------

    if not VAE_CHECKPOINT.is_file():

        raise FileNotFoundError(
            VAE_CHECKPOINT
        )

    for name, path in CHECKPOINTS.items():

        if not Path(path).is_file():

            raise FileNotFoundError(
                "{}: {}".format(name, path)
            )


    # --------------------------------------------------------
    # VERIFY BPM AND MIDI DATASET
    # --------------------------------------------------------

    for bpm in TEMPOS_BPM:

        if bpm not in range(60, 181, 15):

            raise ValueError(
                "Unsupported trained BPM: "
                + str(bpm)
            )

        for pattern in PATTERNS:

            stem = "{}_{}_{}bpm".format(

                PERFORMANCE,

                pattern,

                bpm

            )

            midi_path = (

                DATA_ROOT /

                DATA_SPLIT /

                (stem + ".mid")

            )

            reference_path = (

                DATA_ROOT /

                DATA_SPLIT /

                (stem + ".wav")

            )

            if not midi_path.is_file():

                raise FileNotFoundError(
                    midi_path
                )

            if not reference_path.is_file():

                raise FileNotFoundError(
                    reference_path
                )


    OUTPUT_ROOT.mkdir(

        parents=True,

        exist_ok=True

    )


    # --------------------------------------------------------
    # LOAD PRETRAINED VAE
    # --------------------------------------------------------

    print(
        "\nLoading pretrained VAE..."
    )

    vae = load_vae(

        str(VAE_CHECKPOINT),

        latent_channels=8,

        vae_base_channels=64

    )

    vae.eval()


    # --------------------------------------------------------
    # LOOP OVER CHECKPOINTS
    # --------------------------------------------------------

    for checkpoint_name, checkpoint_path in CHECKPOINTS.items():

        print(
            "\n========================================"
        )

        print(
            "Loading checkpoint:",
            checkpoint_name
        )

        print(
            checkpoint_path
        )


        diffusion = load_diffusion(

            str(checkpoint_path),

            latent_channels=8,

            latent_dim=2

        )

        diffusion.eval()


        checkpoint_output_dir = (

            OUTPUT_ROOT /

            checkpoint_name

        )

        checkpoint_output_dir.mkdir(

            parents=True,

            exist_ok=True

        )


        # ----------------------------------------------------
        # LOOP OVER PATTERNS
        # ----------------------------------------------------

        for pattern in PATTERNS:


            # ------------------------------------------------
            # LOOP OVER BPM VALUES
            # ------------------------------------------------

            for bpm in TEMPOS_BPM:

                stem = "{}_{}_{}bpm".format(

                    PERFORMANCE,

                    pattern,

                    bpm

                )

                midi_path = (

                    DATA_ROOT /

                    DATA_SPLIT /

                    (stem + ".mid")

                )

                reference_path = (

                    DATA_ROOT /

                    DATA_SPLIT /

                    (stem + ".wav")

                )


                filename = (

                    "model_d_30s_"

                    + pattern

                    + "_"

                    + str(bpm)

                    + "bpm_seed"

                    + str(SEED)

                    + ".wav"

                )


                output_path = (

                    checkpoint_output_dir /

                    filename

                )

                metadata_path = (
                    output_path.with_suffix(".json")
                )


                # --------------------------------------------
                # SKIP EXISTING OUTPUTS
                # --------------------------------------------

                if (
                    output_path.exists()
                    or metadata_path.exists()
                ) and not OVERWRITE:

                    if (
                        output_path.exists()
                        and metadata_path.exists()
                    ):

                        print(
                            "Skipping existing file:",
                            filename
                        )

                        continue

                    raise RuntimeError(
                        "Incomplete output exists: "
                        + str(output_path)
                    )


                # --------------------------------------------
                # GENERATE AUDIO
                # --------------------------------------------

                generate_30_seconds(

                    diffusion,

                    vae,

                    bpm,

                    pattern,

                    checkpoint_name,

                    midi_path,

                    reference_path,

                    output_path

                )


        del diffusion

        torch.cuda.empty_cache()


    print(
        "\n========================================"
    )

    print(
        "MODEL D GENERATION COMPLETED"
    )

    print(
        "Output directory:",
        OUTPUT_ROOT
    )

    print(
        "========================================"
    )


# ============================================================
# 7. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
