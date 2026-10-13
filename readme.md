# Model D: BPM and MIDI-onset-conditioned audio diffusion

Model D generates synthetic piano audio conditioned on a requested BPM and a MIDI note-onset pattern. It uses a frozen pretrained audio VAE and a conditional diffusion model in the VAE latent space.

This README documents the synthetic pilot dataset, the relationship between training epochs and optimizer steps, and the reported evaluation of the checkpoint labelled `epoch500`.

## 1. How many audio clips were used for training?

**Model D's diffusion model was trained on 108 audio clips.** A separate set of **36 validation clips** was held out from diffusion training, giving **144 clips in total**.

| Property | Value |
|---|---|
| Training clips | 108 |
| Validation clips | 36 |
| Total clips | 144 |
| Clip duration | 8.192 seconds |
| Samples per clip | 65,536 |
| Sample rate | 8,000 Hz |
| Channels | Stereo |
| Training source performances | 3 |
| Validation source performances | 1 |
| BPM values | 60, 75, 90, 105, 120, 135, 150, 165, 180 |
| Rhythmic patterns | `steady`, `eighth`, `mixed`, `rests` |

The number of clips follows directly from the dataset construction:

```text
Training:   3 source performances × 4 patterns × 9 BPM values = 108 clips
Validation: 1 source performance  × 4 patterns × 9 BPM values =  36 clips
Total:                                                     = 144 clips
```

Each clip has a paired MIDI file. The dataset builder takes note groups from selected MAESTRO MIDI performances, schedules those groups at the chosen BPM and pattern, and renders new piano WAVs using FluidSynth and a SoundFont. These are synthetic rendered recordings.

The builder assigns the BPM, stores it in the MIDI tempo and filename, and records the exact saved MIDI onset times in `manifest.jsonl`. The dataset loader reads the filename BPM and paired MIDI onsets. **Librosa was not used to assign these training labels.**

| Pattern | Scheduling rule |
|---|---|
| `steady` | One attack group per beat |
| `eighth` | Two attack groups per beat, separated by half a beat |
| `mixed` | One attack group per beat, with an extra half-beat attack on beat indices 0, 2, 4, ... |
| `rests` | Skip every fourth beat: indices 3, 7, 11, ... |

The first scheduled attack is at 0.25 seconds. The beat interval is `60 / BPM` seconds. Simultaneous notes in a chord share an onset, so an attack group can contain several notes.

Dataset location:

```text
/data/VAE/maestro_fresh/maestro_tempo_pilot
```

These clip counts describe **Model D diffusion training**. The VAE is loaded from a separate pretrained checkpoint and remains frozen during that training; its pretraining dataset size is not established by this evaluation log.

## 2. How many epochs and steps?

The evaluated checkpoint is labelled **`epoch500`**. For the supplied trainer's default single-GPU configuration, **500 complete epochs correspond to 54,000 optimizer steps**.

**Verification status:** 54,000 is the calculated count under the settings below. The supplied evaluation log does not contain checkpoint training metadata, so it does not independently confirm that this checkpoint completed all 500 epochs. Verify its stored `global_step` and the actual run configuration using the command below.

### Epoch-to-step calculation

An **epoch** is one pass through the training dataset. An **optimizer step** updates the diffusion model's weights after processing the required batch or accumulated batches.

| Setting used for this calculation | Value |
|---|---|
| Training clips | 108 |
| Batch size | 1 |
| Training GPUs | 1 |
| Gradient accumulation | 1 batch per optimizer step |
| Training batches per epoch | 108 |
| Optimizer steps per epoch | 108 |
| Complete epochs being calculated | 500 |
| Calculated optimizer steps | 54,000 |

```text
Training batches per epoch = 108 clips / 1 clip per batch
                           = 108 batches

Optimizer steps per epoch  = 108 batches / 1 batch per update
                           = 108 steps

Optimizer steps in 500 complete epochs
                           = 500 × 108
                           = 54,000 steps
```

The training loader uses `drop_last=True`; with batch size 1, all 108 clips fit into complete batches. This calculation assumes the full loader is processed each epoch with the settings above. Changing batch size, GPU count, gradient accumulation, or the number of processed batches changes the relationship.

| Complete epochs | Optimizer steps at 108 steps/epoch |
|---:|---:|
| 1 | 108 |
| 50 | 5,400 |
| 100 | 10,800 |
| 250 | 27,000 |
| 500 | 54,000 |

Validation batches do not perform optimizer updates and are not included in these counts. Repeatedly processing the same training set does not create additional unique training clips: the unique training set remains 108 clips.

### The step limit matters

The supplied training script defaults to `--max-steps 5000`. Merely setting `--epochs 500` while retaining that step limit does not configure a full 500-epoch run:

```text
5,000 / 108 = 46.2963 epoch equivalents
           = 46 complete epochs plus 32 training batches
```

For a new run intended to complete 500 epochs with this configuration, the relevant arguments are:

```text
--epochs 500 --max-steps -1 --batch-size 1 --devices 1
```

Here `--max-steps -1` disables the separate step cap. These arguments document the intended configuration; no retraining is needed to evaluate or organize the existing checkpoint.

### Verify the existing checkpoint's recorded progress

Run this in the existing environment on the server. It only reads the local checkpoint:

```bash
cd /data/VAE/HarmonAI/VAE-Audio-Diffusion

python3.8 - <<'PY'
import inspect
from pathlib import Path
import torch

path = Path(
    "/data/VAE/HarmonAI/sample-generator-main-1/"
    "model_d_synthetic_500epoch_checkpoints/last.ckpt"
)

kwargs = {"map_location": "cpu"}
if "weights_only" in inspect.signature(torch.load).parameters:
    kwargs["weights_only"] = False
checkpoint = torch.load(str(path), **kwargs)

print("Checkpoint:", path)
print("Stored epoch index:", checkpoint.get("epoch"))
print("Recorded global_step:", checkpoint.get("global_step"))

steps = checkpoint.get("global_step")
if isinstance(steps, int):
    print("Epoch equivalents at 108 steps/epoch:", steps / 108)
    print("Matches calculated 54,000 steps:", steps == 54000)
PY
```

Lightning epoch indices are generally zero-based, so an epoch index of 499 can refer to the 500th epoch. The epoch index alone does not show whether that epoch was complete. Use the recorded step count together with the actual training settings and completion log. Once verified, record those values here as the checkpoint's observed training history.

## 3. Evaluation of the generated audio

### Evaluated set

| Property | Value |
|---|---|
| Checkpoint label | `epoch500` |
| Generated WAVs | 36 |
| Requested duration per WAV | 30 seconds |
| Conditions | 4 patterns × 9 requested BPM values |
| Seed label in filenames | 2026 |
| Evaluator version | `2.0-dataset-schedule` |

The supplied 30-second generator joins 8.192-second generated chunks with crossfades and trims the result to 30 seconds. Its JSON metadata records chunk start times and MIDI onset annotations. The evaluator uses that timeline when constructing the expected attacks.

### Evaluation command

Run from the organized repository with the `dance-diffusion` environment active:

```bash
cd /data/VAE/HarmonAI/VAE-Audio-Diffusion

python3.8 evaluate_model_d_tempo.py \
  --input-dir "/data/VAE/maestro_fresh/model_d_30second_generation_clean/epoch500" \
  --dataset-root "/data/VAE/maestro_fresh/maestro_tempo_pilot" \
  --output-dir "/data/VAE/maestro_fresh/model_d_tempo_evaluation_clean/epoch500"
```

Keep each generation JSON file beside its corresponding WAV.

### Evaluation method

1. **Calibrate attack detection on the original 108 training WAVs.** The detector's threshold and fixed timing delay are selected using their known MIDI onsets. Generated WAVs are not used for calibration.
2. **Audit the 36 held-out validation WAVs.** Check whether the fixed detector and pattern-based tempo estimator recover their known timing.
3. **Detect attacks in generated WAVs.** Compare these detections with the requested MIDI schedule using one-to-one onset precision, recall, and F1 at 50 ms and 100 ms tolerances.
4. **Estimate BPM using the known pattern.** Search candidate tempos from 40 to 240 BPM against the detected attacks and the recorded generation layout. The requested BPM is not passed into this search; it is used afterward to calculate tempo error.

This is a **pattern-conditioned tempo estimate**. For example, the `eighth` pattern explicitly defines two attacks per beat. The evaluator does not infer the rhythm pattern blindly from the audio.

### Detector calibration and reference audit

| Measurement | Reported result |
|---|---:|
| Training WAVs used for calibration | 108 |
| Selected detector threshold (`delta`) | 0.05 |
| Fixed detector delay | 28.0 ms |
| Training onset F1 at 50 ms | 0.997, rounded console value |
| Held-out validation WAVs audited | 36 |
| Validation reference tempos within ±5% | 100.0% — 36/36 |
| Mean validation onset F1 at 50 ms | 0.9980754477872663 |

### Generated-audio results

| Measurement | Reported result |
|---|---:|
| Files evaluated | 36 |
| Files with requested BPM labels | 36 |
| Accepted BPM measurements | 36 |
| Unlabelled files | 0 |
| Labelled but unmeasured files | 0 |
| Mean absolute BPM error | 0.0215277777777807 BPM |
| Median absolute BPM error | 0.025000000000005684 BPM |
| Root mean squared BPM error | 0.03145764348029755 BPM |
| Within ±1 BPM | 100.0% — 36/36 |
| Within ±5% of requested BPM | 100.0% — 36/36 |
| Mean onset F1 at 50 ms | 0.99097989026373 |
| Mean onset F1 at 100 ms | 0.99097989026373 |
| Uncertain tempo estimates | 0 |
| Evaluation errors | 0 |

**Result:** all 36 generated examples satisfied the ±1 BPM and ±5% criteria under this pattern-conditioned evaluator, with mean onset F1 of approximately 0.991 at 50 ms.

BPM error measures agreement between requested and estimated tempo. Onset F1 combines precision and recall for correctly timed detected attacks; F1 = 0.991 is not a statement that 99.1% of every aspect of the audio is correct.

<details>
<summary>Per-file results: all 36 generated examples</summary>

Values below reproduce the rounded console output. Aggregate metrics above were calculated from the evaluator's unrounded values.

| Pattern | Requested BPM | Estimated BPM | Onset F1 at 50 ms |
|---|---:|---:|---:|
| steady | 60 | 59.98 | 0.984 |
| steady | 75 | 75.00 | 1.000 |
| steady | 90 | 90.05 | 0.978 |
| steady | 105 | 105.03 | 0.981 |
| steady | 120 | 120.03 | 1.000 |
| steady | 135 | 135.00 | 1.000 |
| steady | 150 | 150.00 | 0.980 |
| steady | 165 | 165.03 | 1.000 |
| steady | 180 | 180.00 | 1.000 |
| eighth | 60 | 60.00 | 0.992 |
| eighth | 75 | 75.00 | 0.987 |
| eighth | 90 | 90.00 | 1.000 |
| eighth | 105 | 104.97 | 0.990 |
| eighth | 120 | 119.97 | 0.996 |
| eighth | 135 | 135.00 | 0.996 |
| eighth | 150 | 149.95 | 0.990 |
| eighth | 165 | 164.97 | 0.988 |
| eighth | 180 | 179.97 | 0.968 |
| mixed | 60 | 60.00 | 1.000 |
| mixed | 75 | 74.97 | 0.991 |
| mixed | 90 | 90.00 | 1.000 |
| mixed | 105 | 105.00 | 0.994 |
| mixed | 120 | 120.08 | 1.000 |
| mixed | 135 | 134.95 | 0.995 |
| mixed | 150 | 150.03 | 0.991 |
| mixed | 165 | 164.90 | 0.983 |
| mixed | 180 | 180.03 | 0.973 |
| rests | 60 | 59.98 | 1.000 |
| rests | 75 | 75.03 | 1.000 |
| rests | 90 | 90.03 | 0.986 |
| rests | 105 | 105.03 | 0.988 |
| rests | 120 | 120.00 | 0.978 |
| rests | 135 | 135.00 | 0.990 |
| rests | 150 | 149.95 | 0.983 |
| rests | 165 | 165.00 | 0.992 |
| rests | 180 | 180.03 | 1.000 |

All rows had status `measured`. Filenames follow `model_d_30s_<pattern>_<requested_bpm>bpm_seed2026.wav`.

</details>

<details>
<summary>Exact evaluation summary JSON</summary>

```json
{
  "n_files": 36,
  "n_labelled": 36,
  "n_measured": 36,
  "n_unlabelled": 0,
  "n_labelled_unmeasured": 0,
  "n_timing_scored": 36,
  "mae_bpm": 0.0215277777777807,
  "median_absolute_error_bpm": 0.025000000000005684,
  "rmse_bpm": 0.03145764348029755,
  "within_1_bpm_percent_all_labelled": 100.0,
  "within_5_percent_rate_all_labelled": 100.0,
  "mean_onset_f1_50ms": 0.99097989026373,
  "mean_onset_f1_100ms": 0.99097989026373,
  "mean_onset_f1_50ms_all_labelled": 0.99097989026373,
  "n_uncertain_tempo": 0,
  "n_errors": 0
}
```

</details>

### Result files

The evaluation outputs are saved under:

```text
/data/VAE/maestro_fresh/model_d_tempo_evaluation_clean/epoch500
```

| File | Contents |
|---|---|
| `tempo_per_file.csv` | Individual BPM estimates, errors, timing scores, status and flags |
| `tempo_by_condition.csv` | Results grouped by checkpoint, pattern and requested BPM |
| `tempo_summary.json` | Aggregate results, configuration, provenance and per-file records |
| `reference_audit.csv` | Individual held-out validation reference results |
| `reference_summary.json` | Aggregate reference results |
| `detector_calibration.json` | Selected detector settings and calibration provenance |
| `event_details.json` | Expected/detected onset times and generation segments |
| `requested_vs_estimated_bpm.png` | Requested versus estimated BPM plot |
| `requested_vs_estimated_bpm.pdf` | PDF version of the plot |

### Scope of the result

These results support strong timing adherence for the **combined scalar-BPM and MIDI-onset conditioning** in the 36 tested outputs. The evaluator knows the rhythm pattern and generation layout. The onset condition itself also carries tempo information, so these results do not isolate the effect of the scalar-BPM input.

The small fitted BPM error is specific to this estimator and test set. Its fine search spacing is 0.025 BPM; that numerical spacing is not a calibrated perceptual measurement accuracy. The onset detector uses a 4 ms frame hop, and the primary onset matching tolerance is 50 ms.

This run covers one checkpoint and one seed label across the four patterns and nine tempos. It measures rhythmic timing, not audio quality, musical diversity, or clinical effects.

## Implementation and evidence

- `build_maestro_tempo_pilot.py`: dataset size, rendering process and assigned timing rules.
- `dataset/synthetic_maestro_tempo_dataset.py`: BPM label reading and paired MIDI onset loading.
- `train_maestro_onset.py`: batch size, dataset counts and training limits used for the step calculation.
- `generate_model_d_30s.py`: generation settings and chunk metadata.
- `evaluate_model_d_tempo.py`: reference calibration, timing scores and pattern-conditioned BPM estimation.
- The evaluation results above reproduce the user-provided terminal output from the organized repository on 12 October 2026. The checkpoint's stored training-progress metadata remains to be inspected as described in Section 2.
