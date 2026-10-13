from pathlib import Path
from collections import Counter
import re

import pretty_midi
import soundfile as sf
import torch
from torch.utils.data import Dataset


class SyntheticMaestroTempoDataset(Dataset):

    def __init__(
        self,
        base_dir="/data/VAE/maestro_fresh/maestro_tempo_pilot",
        split="train",
        sample_rate=8000,
        sample_size=65536,
        min_bpm=60,
        max_bpm=180,
    ):
        self.sample_rate = sample_rate
        self.sample_size = sample_size

        base = Path(base_dir)
        wavs = sorted((base / split).glob("*.wav"))

        if not wavs:
            raise RuntimeError(f"No WAV files found: {base / split}")

        # Read MIDI onsets and tempo labels for all splits.
        all_records = []

        for wav in sorted(base.rglob("*.wav")):
            if wav.parent.name not in ("train", "validation"):
                continue

            midi_path = wav.with_suffix(".mid")

            if not midi_path.is_file():
                raise FileNotFoundError(midi_path)

            match = re.search(r"_(\d+)bpm$", wav.stem)

            if match is None:
                raise ValueError(f"Missing BPM in filename: {wav}")

            bpm = int(match.group(1))

            if not min_bpm <= bpm <= max_bpm:
                raise ValueError(f"Unexpected BPM: {bpm}")

            midi = pretty_midi.PrettyMIDI(str(midi_path))

            duration = sample_size / sample_rate

            onset_times = sorted(set(
                round(note.start, 6)
                for instrument in midi.instruments
                for note in instrument.notes
                if 0 <= note.start < duration
            ))

            if not onset_times:
                raise ValueError(f"No MIDI onsets: {midi_path}")

            normalized_bpm = (
                2.0 * (bpm - min_bpm)
                / (max_bpm - min_bpm) - 1.0
            )

            all_records.append({
                "wav": wav,
                "midi": midi_path,
                "split": wav.parent.name,
                "bpm": bpm,
                "normalized_bpm": normalized_bpm,
                "onsets": onset_times,
            })

        self.max_onsets = max(
            len(r["onsets"]) for r in all_records
        )

        self.records = [
            r for r in all_records
            if r["split"] == split
        ]

        print(f"Model D {split}: {len(self.records)} clips")
        print("Max MIDI onset slots:", self.max_onsets)
        print(
            "BPM distribution:",
            dict(sorted(Counter(
                r["bpm"] for r in self.records
            ).items()))
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        audio, sr = sf.read(
            str(record["wav"]),
            dtype="float32",
            always_2d=True,
        )

        if sr != self.sample_rate:
            raise ValueError(
                f"Unexpected sample rate: {record['wav']}"
            )

        if audio.shape != (self.sample_size, 2):
            raise ValueError(
                f"Unexpected audio shape: {audio.shape}"
            )

        audio = torch.from_numpy(audio.T.copy())

        if not torch.isfinite(audio).all():
            raise ValueError("Non-finite audio samples")

        tempo = torch.tensor(
            [record["normalized_bpm"]],
            dtype=torch.float32,
        )

        onsets = torch.full(
            (self.max_onsets,),
            -1.0,
            dtype=torch.float32,
        )

        n = len(record["onsets"])

        onsets[:n] = torch.tensor(
            record["onsets"],
            dtype=torch.float32,
        )

        return audio, tempo, onsets, str(record["wav"])
