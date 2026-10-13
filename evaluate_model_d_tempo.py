#!/usr/bin/env python3
"""Model D evaluator aligned with build_maestro_tempo_pilot.py.

Python 3.8+. Install with the Python used to run this script:
  python3.8 -m pip install "librosa==0.10.2.post1" soundfile matplotlib

Example:
  python3.8 evaluate_model_d_tempo.py \
    --input-dir /path/to/generated/epoch500 \
    --dataset-root /data/VAE/maestro_fresh/maestro_tempo_pilot \
    --output-dir /path/to/evaluation/epoch500

The dataset ASSIGNS a BPM, schedules MIDI attacks, and reads BPM from filenames.
It does not estimate training BPM from audio. This script reuses its timing
rules and MIDI annotations, NOT an imaginary inverse of that labeling process.

Primary timing test: detected WAV attacks vs the exact requested MIDI onsets,
with one-to-one precision/recall/F1 at 50 ms and, separately, 100 ms. A fixed
onset-detector threshold and latency are calibrated on ORIGINAL TRAIN WAVs.
Validation reference WAVs are audited separately before generated WAVs.

Supplementary BPM: search the same known pattern's schedule over 40..240 BPM.
The requested BPM is NEVER passed to the search. Known pattern and chunk
layout ARE supplied: this is a custom, pattern-conditioned estimate, not a
blind perceptual beat tracker. An eighth pattern means two attacks per beat.
Poor or ambiguous fits are unmeasured; the best candidate is retained separately.
No octave folding, snapping to requested BPM, or target-dependent prior is used.

30-second layout is read from generator JSON (chunks/start_seconds and
chunk_duration_seconds). Short dataset clips need no JSON. For a long native
continuous schedule, use --layout continuous only if its onset condition was
actually extended using the same rules. Merely making the waveform longer does
not establish the condition's timing. In auto mode an unexplained long layout
is an error. Supplied onset annotations must agree with the assumed layout.

Outputs: tempo_per_file.csv, tempo_by_condition.csv, tempo_summary.json,
reference_audit.csv, reference_summary.json, detector_calibration.json,
event_details.json, and optional PNG/PDF. Results use the actual WAV duration.
Detector calibration and held-out audit are empirical checks, not a guarantee
on generated timbres. Examine F1, failed files and ambiguity flags alongside BPM.
This script does not evaluate sound quality or a clinical outcome.
"""

import argparse
import csv
import hashlib
import json
import math
import platform
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    import librosa
    import numpy as np
    import soundfile as sf
    from numba import njit
except ImportError as exc:
    raise SystemExit('Missing dependency: {}. Run: python3.8 -m pip install '
                     '"librosa==0.10.2.post1" soundfile matplotlib'.format(exc))

VERSION = '2.0-dataset-schedule'
SR, HOP, N_FFT = 8000, 32, 512
CLIP_SECONDS = 65536 / 8000
PATTERNS = ('steady', 'eighth', 'mixed', 'rests')


def make_schedule(bpm, style, duration=CLIP_SECONDS):
    """Same .25-s start, pattern rules and .12-s tail as the dataset builder.

    For training length this is identical to range(64). For an explicitly
    continuous longer layout, extend the beat loop to cover its duration.
    """
    if style not in PATTERNS or not np.isfinite(bpm) or bpm <= 0:
        raise ValueError('Invalid pattern/BPM')
    beat_seconds = 60.0 / bpm
    times = []
    for beat_idx in range(max(64, int(math.ceil(duration / beat_seconds)) + 2)):
        start = 0.25 + beat_idx * beat_seconds
        if start >= duration - 0.12:
            break
        if style == 'rests' and beat_idx % 4 == 3:
            continue
        times.append(start)
        add_half_beat = style == 'eighth' or (style == 'mixed' and beat_idx % 2 == 0)
        if add_half_beat:
            half = start + 0.5 * beat_seconds
            if half < duration - 0.12:
                times.append(half)
    return np.asarray(times, dtype=np.float64)


def timeline(bpm, pattern, segments, duration):
    events = np.concatenate([start + make_schedule(bpm, pattern, length)
                             for start, length in segments])
    return np.unique(events[(events >= 0) & (events < duration)])


def valid_events(value):
    a = np.asarray(value, dtype=float)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all() or np.any(a < 0):
        raise ValueError('Invalid onset annotations')
    # Dataset used sorted unique note starts, rounded to six decimals.
    return np.unique(np.round(a, 6))


def assert_same_events(actual, expected, context):
    if len(actual) != len(expected) or np.max(np.abs(actual - expected), initial=0) > .002:
        raise ValueError('{} does not follow the supplied dataset timing rules. '
                         'Do not score it using an assumed continuous or chunked grid.'
                         .format(context))


def labels_from_name(path):
    match = re.findall(r'(?<![\d.])(\d+(?:\.\d+)?)bpm', path.stem, flags=re.I)
    bpms = set(float(v) for v in match)
    patterns = [p for p in PATTERNS if re.search(r'(?:^|_)' + p + r'(?:_|$)', path.stem)]
    if len(bpms) > 1 or len(patterns) > 1:
        raise ValueError('Ambiguous labels in filename: ' + path.name)
    return (next(iter(bpms)) if bpms else None, patterns[0] if patterns else None)


def target_and_pattern(path, meta):
    named_bpm, named_pattern = labels_from_name(path)
    values = [float(meta[k]) for k in ('tempo_bpm', 'requested_bpm', 'target_bpm')
              if meta.get(k) is not None]
    if not values and meta.get('scalar_bpm') is not None:
        values.append(float(meta['scalar_bpm']))
    if named_bpm is not None:
        values.append(named_bpm)
    if not values or not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError('Missing or invalid requested BPM in JSON/filename')
    if max(values) - min(values) > .1:
        raise ValueError('Conflicting BPM labels in JSON/filename')
    pattern = meta.get('pattern') or meta.get('style') or named_pattern
    if pattern not in PATTERNS or (named_pattern and pattern != named_pattern):
        raise ValueError('Missing, unsupported or conflicting pattern label')
    # Scalar/grid conflict is a conditioning ablation, not a single target.
    if meta.get('grid_bpm') is not None and abs(float(meta['grid_bpm']) - values[0]) > .1:
        raise ValueError('Scalar/grid ablation requires separate targets; this report expects coupled conditioning')
    return values[0], pattern


def load_audio(path):
    audio, sr = sf.read(str(path), dtype='float32', always_2d=True)
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError('Empty or nonfinite audio: ' + str(path))
    power = np.mean(audio.astype(np.float64) ** 2, axis=0)
    channel = int(np.argmax(power))
    y = audio[:, channel].copy()
    y -= np.mean(y)
    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
    info = dict(duration_seconds=len(audio) / float(sr), native_sample_rate=int(sr),
                selected_channel=channel, rms_dbfs=20 * math.log10(max(rms, 1e-12)))
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    if rms < 10 ** (-70 / 20):
        return np.zeros(1, dtype=float), info
    y /= max(float(np.max(np.abs(y))), 1e-12)
    env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP, n_fft=N_FFT)
    return env, info


def detect(env, delta, delay=0.0):
    if not len(env) or np.max(env) <= 1e-10:
        return np.empty(0, dtype=np.float64)
    times = librosa.onset.onset_detect(
        onset_envelope=env, sr=SR, hop_length=HOP, units='time',
        backtrack=False, normalize=True, delta=delta,
        pre_max=5, post_max=5, pre_avg=25, post_avg=25, wait=15)
    return np.asarray(times, dtype=np.float64) - delay


@njit
def match_stats(detected, expected, tol, shift=0.0):
    """Maximum-cardinality one-to-one matching of sorted events at tolerance.

    Greedy matching is sufficient for this ordered, fixed-radius problem.
    The soft score breaks tempo-grid ties; it is a custom fitting objective.
    """
    i = j = hits = 0
    soft = 0.0
    absolute = 0.0
    while i < len(detected) and j < len(expected):
        difference = detected[i] - (expected[j] + shift)
        if abs(difference) <= tol:
            hits += 1
            absolute += abs(difference)
            soft += math.exp(-0.5 * (difference / (tol / 2)) ** 2)
            i += 1
            j += 1
        elif difference < 0:
            i += 1
        else:
            j += 1
    denominator = len(detected) + len(expected)
    return (hits, 2.0 * hits / denominator if denominator else 0.0,
            2.0 * soft / denominator if denominator else 0.0,
            absolute / hits if hits else -1.0)


def onset_metrics(detected, expected, tol=.05):
    hits, f1, soft, error = match_stats(detected, expected, tol)
    return dict(expected_onsets=len(expected), detected_onsets=len(detected), matched_onsets=int(hits),
                precision=hits / len(detected) if len(detected) else 0.0,
                recall=hits / len(expected) if len(expected) else 0.0, f1=f1,
                matched_onset_mae_ms=1000 * error if hits else None)


@njit
def scan_templates(detected, templates, lengths, offsets):
    scores = np.empty(len(templates))
    shifts = np.empty(len(templates))
    for i in range(len(templates)):
        best = -1.0
        best_offset = 0.0
        for offset in offsets:
            _, _, value, _ = match_stats(detected, templates[i, :lengths[i]], .05, offset)
            if value > best:
                best = value
                best_offset = offset
        scores[i] = best
        shifts[i] = best_offset
    return scores, shifts


def template_batch(bpms, pattern, segments, duration):
    ts = [timeline(b, pattern, segments, duration) for b in bpms]
    lengths = np.asarray([len(t) for t in ts], dtype=np.int64)
    padded = np.zeros((len(ts), max(lengths)), dtype=np.float64)
    for i, t in enumerate(ts):
        padded[i, :len(t)] = t
    return padded, lengths


def estimate_pattern_bpm(detected, pattern, segments, duration, min_bpm=40.0, max_bpm=240.0):
    """Deliberately has NO requested BPM or requested onset list argument."""
    result = dict(estimated_bpm=None, best_candidate_bpm=None, fit_soft_score=None,
                  fit_f1_50ms=None, fit_offset_seconds=None, alternative_bpm=None,
                  alternative_soft_score=None, fit_margin=None, status='unmeasured', flags=[])
    if len(detected) < 6 or duration < 4:
        result['status'] = 'too_few_onsets_or_too_short'
        return result
    # Full, target-independent search. Refine eight distinct high-scoring modes
    # so an alias is not rejected merely because its coarse-grid alignment is bad.
    bpms = np.arange(min_bpm, max_bpm + .001, .5)
    templates, lengths = template_batch(bpms, pattern, segments, duration)
    offsets = np.arange(-.12, .1201, .01)
    scores, shifts = scan_templates(detected, templates, lengths, offsets)
    selected = []
    for idx in np.argsort(-scores):
        if all(abs(bpms[idx] - bpms[j]) > 2 for j in selected):
            selected.append(int(idx))
        if len(selected) == 8:
            break
    fine_bpms = np.unique(np.round(np.concatenate([
        np.arange(max(min_bpm, bpms[j] - .55), min(max_bpm, bpms[j] + .55) + .001, .025)
        for j in selected]), 6))
    fine_templates, fine_lengths = template_batch(fine_bpms, pattern, segments, duration)
    fine_scores, fine_shifts = scan_templates(
        detected, fine_templates, fine_lengths, np.arange(-.12, .1201, .004))
    all_bpms = np.concatenate((bpms, fine_bpms))
    all_scores = np.concatenate((scores, fine_scores))
    all_shifts = np.concatenate((shifts, fine_shifts))
    best = int(np.argmax(all_scores))
    bpm, score = float(all_bpms[best]), float(all_scores[best])
    shift = float(all_shifts[best])
    other = np.flatnonzero(np.abs(all_bpms - bpm) / bpm > .05)
    rival = int(other[np.argmax(all_scores[other])]) if len(other) else None
    alt_score = float(all_scores[rival]) if rival is not None else 0.0
    fitted = timeline(bpm, pattern, segments, duration)
    _, f1, _, _ = match_stats(detected, fitted, .05, shift)
    result.update(best_candidate_bpm=bpm, fit_soft_score=score, fit_f1_50ms=f1,
                  fit_offset_seconds=shift, alternative_bpm=float(all_bpms[rival]) if rival is not None else None,
                  alternative_soft_score=alt_score, fit_margin=score - alt_score)
    flags = result['flags']
    if score < .65 or f1 < .70:
        flags.append('poor_pattern_fit')
    if score - alt_score < .03:
        flags.append('ambiguous_pattern_tempo')
    if bpm <= min_bpm + .5 or bpm >= max_bpm - .5:
        flags.append('at_search_boundary')
    if flags:
        result['status'] = 'uncertain_tempo'
    else:
        result.update(status='measured', estimated_bpm=bpm)
    if abs(shift) >= .10:
        flags.append('large_timing_offset')
    return result


def resolve_layout(meta, duration, layout):
    if layout == 'continuous':
        return [(0.0, duration)], 'explicit_continuous'
    chunks = meta.get('chunks')
    if chunks:
        length = float(meta.get('chunk_duration_seconds', 0))
        if not math.isfinite(length) or length <= 0:
            raise ValueError('Chunk metadata needs chunk_duration_seconds')
        starts = [float(c['start_seconds']) for c in chunks]
        if (not all(math.isfinite(s) and 0 <= s < duration for s in starts)
                or starts != sorted(set(starts)) or abs(starts[0]) > .001):
            raise ValueError('Invalid chunk start times')
        if starts[-1] + length < duration - .002:
            raise ValueError('Chunk metadata does not cover WAV duration')
        if any(b > a + length + .002 for a, b in zip(starts, starts[1:])):
            raise ValueError('Chunk metadata has uncovered gaps')
        return [(s, length) for s in starts], 'metadata_chunks'
    if duration <= CLIP_SECONDS + .002:
        # Do not shorten the MIDI schedule merely because a WAV was cropped.
        return [(0.0, CLIP_SECONDS)], 'dataset_clip'
    # Recognize an explicitly annotated global continuous schedule. Verification
    # against these annotations happens after the independent layout decision.
    if meta.get('midi_onsets_seconds') is not None:
        return [(0.0, duration)], 'annotated_continuous'
    raise ValueError('Long WAV has no supported timing layout. Keep the generator JSON beside it. '
                     'Use --layout continuous ONLY for a genuinely continuous extended onset condition.')


def expected_events(meta, bpm, pattern, segments, duration, mode):
    analytic = timeline(bpm, pattern, segments, duration)
    if meta.get('midi_onsets_per_chunk_seconds') is not None:
        if mode != 'metadata_chunks':
            raise ValueError('Per-chunk annotations conflict with non-chunked layout')
        local = valid_events(meta['midi_onsets_per_chunk_seconds'])
        assert_same_events(local, make_schedule(bpm, pattern, segments[0][1]), 'Chunk MIDI')
        expected = np.unique(np.concatenate([local + s for s, _ in segments]))
        expected = expected[(expected >= 0) & (expected < duration)]
        assert_same_events(expected, analytic, 'Expanded MIDI')
        return expected, 'generation_json_midi_onsets'
    if meta.get('midi_onsets_seconds') is not None:
        expected = valid_events(meta['midi_onsets_seconds'])
        expected = expected[expected < duration]
        assert_same_events(expected, analytic, 'Global MIDI')
        return expected, 'generation_json_midi_onsets'
    return analytic, 'reconstructed_dataset_schedule'


def load_references(root):
    manifest = root / 'manifest.jsonl'
    if not manifest.is_file():
        raise FileNotFoundError('Dataset manifest not found: ' + str(manifest))
    records, seen = [], set()
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r['split'] not in ('train', 'validation'):
            continue
        original = Path(r['audio_path'])
        # Prefer the explicitly supplied dataset root over stale absolute paths.
        path = root / r['split'] / original.name
        if not path.is_file():
            raise FileNotFoundError(path)
        bpm, pattern = labels_from_name(path)
        if bpm != float(r['bpm']) or pattern != r['style']:
            raise ValueError('Manifest/filename disagreement: ' + str(path))
        key = str(path.resolve())
        if key in seen:
            raise ValueError('Duplicate reference record: ' + key)
        seen.add(key)
        expected = valid_events(r['onsets'])
        assert_same_events(expected, make_schedule(bpm, pattern), 'Manifest MIDI ' + path.name)
        records.append(dict(path=path, split=r['split'], bpm=bpm, pattern=pattern, expected=expected))
    for split in ('train', 'validation'):
        if not any(r['split'] == split for r in records):
            raise ValueError('Manifest needs both train and validation references')
    return records, manifest


def calibrate(records):
    train = [r for r in records if r['split'] == 'train']
    deltas = (.05, .10, .15, .20, .25)
    detected = []
    print('Calibrating on {} original TRAIN WAVs; generated audio is not used.'.format(len(train)), flush=True)
    for i, r in enumerate(train, 1):
        env, info = load_audio(r['path'])
        if abs(info['duration_seconds'] - CLIP_SECONDS) > .002 or info['native_sample_rate'] != SR:
            raise ValueError('Reference format differs from dataset builder: ' + str(r['path']))
        detected.append([detect(env, d) for d in deltas])
        if i == 1 or i % 12 == 0 or i == len(train):
            print('  Reference features {}/{}'.format(i, len(train)), flush=True)
    choices = []
    for di, delta in enumerate(deltas):
        for delay in np.arange(-.040, .0801, .004):
            scores = [match_stats(ds[di] - delay, r['expected'], .05) for ds, r in zip(detected, train)]
            choices.append((float(np.mean([s[1] for s in scores])),
                            float(np.mean([s[2] for s in scores])), -abs(float(delay)), delta, float(delay)))
    best = max(choices)
    return dict(delta=best[3], detector_delay_seconds=best[4], calibration_macro_f1_50ms=best[0],
                calibration_soft_score=best[1], n_training_references=len(train),
                analysis_sr=SR, hop_length=HOP, n_fft=N_FFT,
                threshold_candidates=list(deltas), latency_search_seconds=[-.04, .08],
                latency_step_seconds=.004, match_tolerance_seconds=.05,
                selection='Maximize training-reference macro F1, then soft timing score; never tune on generated audio',
                peak_picker=dict(pre_max=5, post_max=5, pre_avg=25, post_avg=25, wait=15),
                training_files=[str(r['path']) for r in train])


def add_measurements(row, detected, expected, pattern, segments, duration, args):
    for tol, suffix in ((.05, '50ms'), (.1, '100ms')):
        m = onset_metrics(detected, expected, tol)
        row.update({'onset_' + k + '_' + suffix: v for k, v in m.items()})
    # Dense patterns get nonzero accidental matches. This fixed phase-shift
    # control is diagnostic; it is not a calibrated significance test.
    shifted = []
    for shift in (.137, .281, .419, .613):
        null = np.sort(np.mod(detected + shift, duration))
        shifted.append(onset_metrics(null, expected, .05)['f1'])
    row['shifted_control_median_f1_50ms'] = float(np.median(shifted))
    row['f1_minus_shifted_control'] = row['onset_f1_50ms'] - row['shifted_control_median_f1_50ms']
    fit = estimate_pattern_bpm(detected, pattern, segments, duration, args.min_bpm, args.max_bpm)
    row.update(fit)
    target = row['requested_bpm']
    estimate = row['estimated_bpm']
    row.update(absolute_error_bpm=None, signed_error_bpm=None, absolute_percentage_error=None,
               within_1_bpm=False, within_5_percent=False)
    if estimate is not None:
        error = estimate - target
        row.update(signed_error_bpm=error, absolute_error_bpm=abs(error),
                   absolute_percentage_error=100 * abs(error) / target,
                   within_1_bpm=abs(error) <= 1, within_5_percent=abs(error) / target <= .05)
    row['flags'] = ';'.join(row['flags'])


def summarize(rows):
    labelled = [r for r in rows if r.get('requested_bpm') is not None]
    measured = [r for r in labelled if r.get('estimated_bpm') is not None]
    timing = [r for r in labelled if r.get('onset_f1_50ms') is not None]
    errors = [r['absolute_error_bpm'] for r in measured]
    return dict(n_files=len(rows), n_labelled=len(labelled), n_measured=len(measured),
                n_unlabelled=len(rows) - len(labelled),
                n_labelled_unmeasured=len(labelled) - len(measured),
                n_timing_scored=len(timing),
                mae_bpm=float(np.mean(errors)) if errors else None,
                median_absolute_error_bpm=float(np.median(errors)) if errors else None,
                rmse_bpm=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
                within_1_bpm_percent_all_labelled=100 * sum(r.get('within_1_bpm', False) for r in labelled) / len(labelled) if labelled else None,
                within_5_percent_rate_all_labelled=100 * sum(r.get('within_5_percent', False) for r in labelled) / len(labelled) if labelled else None,
                mean_onset_f1_50ms=float(np.mean([r['onset_f1_50ms'] for r in timing])) if timing else None,
                mean_onset_f1_100ms=float(np.mean([r['onset_f1_100ms'] for r in timing])) if timing else None,
                mean_onset_f1_50ms_all_labelled=sum(r.get('onset_f1_50ms', 0) for r in labelled) / len(labelled) if labelled else None,
                n_uncertain_tempo=sum(r.get('status') == 'uncertain_tempo' for r in rows),
                n_errors=sum(r.get('status') == 'error' for r in rows))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    valid = [r for r in rows if r.get('estimated_bpm') is not None]
    if not valid:
        return 'No accepted estimates; no plot.'
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    high = max(max(r['requested_bpm'], r['estimated_bpm']) for r in valid) * 1.1
    grid = np.linspace(0, high, 200)
    ax.fill_between(grid, .95 * grid, 1.05 * grid, alpha=.12, color='green', label='±5%')
    ax.plot(grid, grid, color='gray', linewidth=1, label='Perfect agreement')
    for pattern in PATTERNS:
        selected = [r for r in valid if r['pattern'] == pattern]
        if selected:
            ax.scatter([r['requested_bpm'] for r in selected], [r['estimated_bpm'] for r in selected], label=pattern, alpha=.7)
    ax.set(xlabel='Requested BPM', ylabel='Pattern-conditioned estimated BPM',
           xlim=(0, high), ylim=(0, high), title='Model D: dataset-pattern tempo estimate\n{} / {} files measured'.format(len(valid), len(rows)))
    ax.set_aspect('equal')
    ax.grid(alpha=.2)
    ax.legend()
    for ext in ('png', 'pdf'):
        fig.savefig(output / ('requested_vs_estimated_bpm.' + ext), dpi=300)
    plt.close(fig)
    return 'Created PNG and PDF. Uncertain files excluded; see counts/CSV.'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input-dir', '--input', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--dataset-root', type=Path, default=Path('/data/VAE/maestro_fresh/maestro_tempo_pilot'))
    p.add_argument('--layout', choices=('auto', 'continuous'), default='auto')
    p.add_argument('--min-bpm', type=float, default=40.0)
    p.add_argument('--max-bpm', type=float, default=240.0)
    p.add_argument('--no-plots', action='store_true')
    a = p.parse_args()
    if not (10 <= a.min_bpm < a.max_bpm <= 300) or a.max_bpm - a.min_bpm < 60:
        p.error('Use a broad search range: 10 <= min-bpm < max-bpm <= 300; width at least 60 BPM')
    source = a.input_dir.expanduser().resolve()
    if not source.exists():
        p.error('Input not found: ' + str(source))
    files = [source] if source.is_file() else sorted(f for f in source.rglob('*') if f.is_file() and f.suffix.lower() == '.wav')
    if not files or any(f.suffix.lower() != '.wav' for f in files):
        p.error('No WAV files found')
    output = a.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    records, manifest = load_references(a.dataset_root.expanduser().resolve())
    calibration = calibrate(records)
    write_json(output / 'detector_calibration.json', calibration)
    print('Fixed detector: delta={:.2f}, delay={:.1f} ms, training F1={:.3f}'.format(
        calibration['delta'], 1000 * calibration['detector_delay_seconds'], calibration['calibration_macro_f1_50ms']), flush=True)
    refs = [r for r in records if r['split'] == 'validation']
    reference_rows = []
    print('Auditing {} held-out VALIDATION references...'.format(len(refs)), flush=True)
    for i, r in enumerate(refs, 1):
        row = dict(file=str(r['path']), pattern=r['pattern'], requested_bpm=r['bpm'], status='error', error='')
        try:
            env, info = load_audio(r['path'])
            if abs(info['duration_seconds'] - CLIP_SECONDS) > .002 or info['native_sample_rate'] != SR:
                raise ValueError('Validation reference format differs from dataset builder')
            detected = detect(env, calibration['delta'], calibration['detector_delay_seconds'])
            row.update(info)
            add_measurements(row, detected, r['expected'], r['pattern'], [(0., CLIP_SECONDS)], CLIP_SECONDS, a)
        except Exception as exc:
            row.update(status='error', estimated_bpm=None, error='{}: {}'.format(type(exc).__name__, exc))
        reference_rows.append(row)
        print('  Reference {}/{} {} | {}'.format(i, len(refs), r['path'].name, row['status']), flush=True)
    ref_summary = summarize(reference_rows)
    write_csv(output / 'reference_audit.csv', reference_rows)
    write_json(output / 'reference_summary.json', ref_summary)
    reference_groups = {pattern: summarize([r for r in reference_rows if r['pattern'] == pattern]) for pattern in PATTERNS}
    print('Reference direct ±5% rate: {}; mean F1(50 ms): {}'.format(
        ref_summary['within_5_percent_rate_all_labelled'], ref_summary['mean_onset_f1_50ms']), flush=True)
    rows, details = [], {}
    print('Evaluating generated WAVs...', flush=True)
    for i, path in enumerate(files, 1):
        row = dict(file=str(path), checkpoint=path.parent.name, pattern='unknown', requested_bpm=None,
                   estimated_bpm=None, status='error', error='')
        try:
            sidecar = path.with_suffix('.json')
            meta = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
            if not isinstance(meta, dict):
                raise ValueError('JSON sidecar must be an object')
            target, pattern = target_and_pattern(path, meta)
            row.update(requested_bpm=target, pattern=pattern, checkpoint=meta.get('checkpoint_name', path.parent.name))
            env, info = load_audio(path)
            duration = info['duration_seconds']
            row.update(info)
            segments, layout = resolve_layout(meta, duration, a.layout)
            row['timing_layout'] = layout
            expected, origin = expected_events(meta, target, pattern, segments, duration, layout)
            row['expected_onsets_source'] = origin
            detected = detect(env, calibration['delta'], calibration['detector_delay_seconds'])
            detected = detected[(detected >= 0) & (detected < duration)]
            add_measurements(row, detected, expected, pattern, segments, duration, a)
            baseline = reference_groups[pattern]
            row['reference_mean_f1_50ms_for_pattern'] = baseline['mean_onset_f1_50ms']
            row['reference_within_5_percent_for_pattern'] = baseline['within_5_percent_rate_all_labelled']
            baseline_good = (baseline['n_files'] > 0 and baseline['n_errors'] == 0
                             and (baseline['mean_onset_f1_50ms'] or 0) >= .80
                             and (baseline['within_5_percent_rate_all_labelled'] or 0) >= 90)
            row['reference_audit_passes_heuristic'] = baseline_good
            if not baseline_good:
                row['flags'] = ';'.join(filter(None, [row['flags'], 'reference_audit_weak_for_pattern']))
            details[str(path)] = dict(expected_onsets_seconds=expected.tolist(),
                                      detected_onsets_seconds=detected.tolist(), segments=segments)
        except Exception as exc:
            row.update(status='error', estimated_bpm=None, error='{}: {}'.format(type(exc).__name__, exc))
        rows.append(row)
        estimate = 'unmeasured' if row.get('estimated_bpm') is None else '{:.2f}'.format(row['estimated_bpm'])
        f1 = '-' if row.get('onset_f1_50ms') is None else '{:.3f}'.format(row['onset_f1_50ms'])
        print('[{}/{}] {} | requested={} estimated={} | onset F1(50ms)={} | {} {}'.format(
            i, len(files), path.name, row['requested_bpm'], estimate, f1, row['status'], row.get('flags') or row['error']), flush=True)
    groups = defaultdict(list)
    for row in rows:
        groups[(row['checkpoint'], row['pattern'], row['requested_bpm'])].append(row)
    grouped = [dict(checkpoint=k[0], pattern=k[1], requested_bpm=k[2], **summarize(v))
               for k, v in sorted(groups.items(), key=lambda kv: repr(kv[0]))]
    summary = summarize(rows)
    write_csv(output / 'tempo_per_file.csv', rows)
    write_csv(output / 'tempo_by_condition.csv', grouped)
    write_json(output / 'event_details.json', details)
    report = dict(evaluator_version=VERSION, created_utc=datetime.now(timezone.utc).isoformat(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                  versions=dict(python=platform.python_version(), librosa=librosa.__version__, numpy=np.__version__, soundfile=sf.__version__),
                  settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                  summary=summary, reference_summary=ref_summary, reference_by_pattern=reference_groups,
                  calibration=calibration,
                  method='Assigned MIDI labels; exact schedule onset agreement; supplementary target-independent, known-pattern BPM search',
                  estimate_acceptance_heuristics=dict(min_soft_score=.65, min_f1_50ms=.70, min_margin_outside_5_percent=.03),
                  reference_audit_heuristics=dict(min_mean_f1_50ms=.80, min_within_5_percent_rate=90),
                  denominators='MAE/RMSE use accepted labelled estimates. BPM success rates and all-labelled F1 include failed labelled files. See unlabelled count separately.',
                  limitations='Custom pattern-conditioned estimator, not validated perceptual tempo. Pattern/timeline are supplied. Missing attacks can create tempo aliases; F1 and reference audit must accompany BPM. Acceptance thresholds are heuristics, not calibrated confidence.',
                  sources=['User-supplied build_maestro_tempo_pilot.py and SyntheticMaestroTempoDataset',
                           'https://librosa.org/doc/0.10.2/generated/librosa.onset.onset_detect.html',
                           'https://mir-eval.readthedocs.io/stable/api/onset.html'], files=rows)
    for ext in ('png', 'pdf'):
        prior_plot = output / ('requested_vs_estimated_bpm.' + ext)
        if prior_plot.exists():
            prior_plot.unlink()
    # Clear the old evaluator's window CSV when reusing its output directory.
    stale_windows = output / 'tempo_windows.csv'
    if stale_windows.exists():
        stale_windows.unlink()
    if not a.no_plots:
        try:
            report['plot_status'] = plot(rows, output)
        except Exception as exc:
            report['plot_status'] = 'Plot failed: {}'.format(exc)
    write_json(output / 'tempo_summary.json', report)
    print('\n' + json.dumps(summary, indent=2))
    print('\nResults:', output)
    if not all(r.get('reference_audit_passes_heuristic', False) for r in rows):
        print('Check reference_audit.csv and per-file flags before interpreting BPM scores.', file=sys.stderr)
    if not summary['n_timing_scored']:
        raise SystemExit('No files scored. Read tempo_per_file.csv errors.')


if __name__ == '__main__':
    main()
