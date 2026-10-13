
import torch


def make_scalar_onset_condition(
    tempo_vec,
    onset_times,
    latent_length,
    sample_rate=8000,
    downsample_factor=16,
    onset_sigma=0.03,
):
    """
    tempo_vec: [B, 1], normalized BPM
    onset_times: [B, N], MIDI onset timestamps in seconds
                 Pad shorter sequences with -1.
    latent_length: number of latent time samples

    Returns:
        [B, 2, latent_length]

        Channel 0: normalized BPM
        Channel 1: MIDI-derived onset pulses
    """

    if tempo_vec.ndim != 2 or tempo_vec.shape[1] != 1:
        raise ValueError("tempo_vec must have shape [B, 1]")

    if onset_times.ndim != 2:
        raise ValueError("onset_times must have shape [B, N]")

    if onset_times.shape[0] != tempo_vec.shape[0]:
        raise ValueError("Batch sizes must match")

    if onset_times.shape[1] == 0:
        raise ValueError("Expected at least one onset slot")

    if onset_sigma <= 0:
        raise ValueError("onset_sigma must be positive")

    onset_times = onset_times.to(
        device=tempo_vec.device,
        dtype=tempo_vec.dtype
    )

    # Position of each latent sample in seconds
    times = torch.arange(
        latent_length,
        device=tempo_vec.device,
        dtype=tempo_vec.dtype
    )

    times = (
        times * downsample_factor / sample_rate
    ).view(1, 1, -1)

    # [B, N, 1]
    events = onset_times.unsqueeze(-1)

    # Ignore padded onset timestamps
    valid = events >= 0

    # Gaussian pulse around every MIDI note onset
    pulses = torch.exp(
        -0.5 * ((times - events) / onset_sigma) ** 2
    )

    pulses = pulses * valid.to(pulses.dtype)

    # Combine note events into one timing channel
    onset_grid = pulses.amax(dim=1)

    # Scalar BPM channel
    scalar_grid = tempo_vec.unsqueeze(-1).expand(
        -1, -1, latent_length
    )

    condition = torch.cat(
        [scalar_grid, onset_grid.unsqueeze(1)],
        dim=1
    )

    return condition