#data preprocessing
import csv
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

BASE = Path("/data/VAE/maestro_fresh")
ROOT = BASE / "maestro-v3.0.0"
OUT = BASE / "maestro_tempo_pilot"

SYNTH = BASE / "renderer_env_v2/bin/fluidsynth"
SFONT = BASE / "soundfonts/MuseScore_General.sf3"

SR = 8000
SAMPLES = 65536
DURATION = SAMPLES / SR
BPMS = list(range(60, 181, 15))
STYLES = ["steady", "eighth", "mixed", "rests"]

rng = random.Random(2026)

with open(ROOT / "maestro-v3.0.0.csv", newline="") as f:
    metadata = list(csv.DictReader(f))


def get_groups(midi_path):
    source = pretty_midi.PrettyMIDI(str(midi_path))
    notes = sorted(
        (note for inst in source.instruments
         if not inst.is_drum for note in inst.notes),
        key=lambda note: note.start
    )

    groups = []
    for note in notes:
        if not groups or note.start - groups[-1][0].start > 0.030:
            groups.append([note])
        else:
            groups[-1].append(note)

    return groups


def make_schedule(bpm, style):
    beat_seconds = 60.0 / bpm
    times = []

    for beat_idx in range(64):
        start = 0.25 + beat_idx * beat_seconds
        if start >= DURATION - 0.12:
            break

        if style == "rests" and beat_idx % 4 == 3:
            continue

        times.append(start)

        add_half_beat = (
            style == "eighth" or
            (style == "mixed" and beat_idx % 2 == 0)
        )

        if add_half_beat:
            half = start + 0.5 * beat_seconds
            if half < DURATION - 0.12:
                times.append(half)

    return times


selected = []

for split, count in [("train", 3), ("validation", 1)]:
    available = [r for r in metadata if r["split"] == split]
    selected.extend(rng.sample(available, count))

records = []

for performance_idx, row in enumerate(selected):
    groups = get_groups(ROOT / row["midi_filename"])

    if len(groups) < 100:
        raise RuntimeError("Too few groups: " + row["midi_filename"])

    # Select a passage away from the performance boundaries.
    offset = rng.randint(5, len(groups) - 65)
    material = groups[offset:offset + 60]

    split = row["split"]
    split_dir = OUT / split
    split_dir.mkdir(parents=True, exist_ok=True)

    print("\nPerformance:", performance_idx, split)
    print("Selected source groups:", len(material))

    for style in STYLES:
        for bpm in BPMS:
            schedule = make_schedule(bpm, style)

            midi = pretty_midi.PrettyMIDI(
                initial_tempo=bpm, resolution=960
            )
            piano = pretty_midi.Instrument(program=0)

            for idx, onset in enumerate(schedule):
                group = material[idx]
                beat_seconds = 60.0 / bpm

                for original in group:
                    note_length = min(
                        max(
                            original.end - original.start,
                            0.25 * beat_seconds
                        ),
                        0.85 * beat_seconds,
                        DURATION - 0.01 - onset
                    )

                    if note_length <= 0:
                        continue

                    piano.notes.append(pretty_midi.Note(
                        pitch=original.pitch,
                        velocity=original.velocity,
                        start=onset,
                        end=onset + note_length
                    ))

            midi.instruments.append(piano)

            stem = (
                f"performance{performance_idx:02d}_"
                f"{style}_{bpm}bpm"
            )

            midi_path = split_dir / (stem + ".mid")
            raw_path = split_dir / (stem + "_raw.wav")
            wav_path = split_dir / (stem + ".wav")

            midi.write(str(midi_path))

            subprocess.run(
                [
                    str(SYNTH), "-ni", "-g", "0.4",
                    "-F", str(raw_path), "-r", str(SR),
                    str(SFONT), str(midi_path)
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            audio, sample_rate = sf.read(
                raw_path,
                dtype="float32",
                always_2d=True
            )

            assert sample_rate == SR
            assert audio.shape[1] == 2

            audio = audio[:SAMPLES]

            if len(audio) < SAMPLES:
                audio = np.pad(
                    audio,
                    ((0, SAMPLES - len(audio)), (0, 0))
                )

            audio[-160:] *= np.linspace(
                1, 0, 160, dtype=np.float32
            )[:, None]

            peak = float(np.max(np.abs(audio)))
            rms = float(np.sqrt(np.mean(audio ** 2)))

            if not np.isfinite(audio).all():
                raise RuntimeError("Nonfinite audio: " + stem)
            if peak >= 0.99 or rms < 0.001:
                raise RuntimeError("Audio quality check failed: " + stem)

            sf.write(wav_path, audio, SR, subtype="PCM_16")
            raw_path.unlink()

            # Read back MIDI to capture its exact stored timestamps.
            saved_midi = pretty_midi.PrettyMIDI(str(midi_path))
            onsets = sorted({
                round(n.start, 6)
                for inst in saved_midi.instruments
                for n in inst.notes
            })

            records.append({
                "split": split,
                "source_midi": row["midi_filename"],
                "bpm": bpm,
                "style": style,
                "audio_path": str(wav_path),
                "midi_path": str(midi_path),
                "onsets": onsets,
                "note_count": len(piano.notes),
                "rms": round(rms, 6),
            })

            print(
                f"{stem}: {len(onsets)} onsets, "
                f"RMS={rms:.4f}"
            )

manifest = OUT / "manifest.jsonl"
with open(manifest, "w") as f:
    for rec in records:
        f.write(json.dumps(rec) + "\n")

print("\nCompleted:", len(records), "recordings")
print("Training:", sum(r["split"] == "train" for r in records))
print("Validation:", sum(r["split"] == "validation" for r in records))
print("Manifest:", manifest)
