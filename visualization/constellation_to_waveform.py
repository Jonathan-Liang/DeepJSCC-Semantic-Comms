import argparse
import csv
import json
import os

import numpy as np


def get_parser():
    """Create parser for constellation-to-waveform conversion options."""
    parser = argparse.ArgumentParser(description="Convert I/Q constellation points into a pulse-shaped complex waveform.")
    parser.add_argument("--points_csv", type=str, required=True, help="CSV created by plot_encoder_constellation.py --save_points_csv")
    parser.add_argument("--pair", type=int, default=None, help="Optional pair index filter (default: include all pairs)")
    parser.add_argument("--max_symbols", type=int, default=None, help="Optional max number of symbols to use")
    parser.add_argument("--sps", type=int, default=8, help="Samples per symbol")
    parser.add_argument("--rrc_beta", type=float, default=0.25, help="RRC rolloff factor beta in [0, 1]")
    parser.add_argument("--rrc_span", type=int, default=10, help="RRC span in symbols")
    parser.add_argument("--target_power", type=float, default=1.0, help="Target average complex baseband power")
    parser.add_argument("--clip_mag", type=float, default=None, help="Optional magnitude clip threshold after normalization")
    parser.add_argument("--gain", type=float, default=1.0, help="Final scalar gain")
    parser.add_argument("--sample_rate", type=float, default=1.0, help="Sample rate in Hz for optional frequency offset")
    parser.add_argument("--freq_offset_hz", type=float, default=0.0, help="Optional digital frequency shift in Hz")
    parser.add_argument("--output_dir", type=str, default="./outputs/constellation", help="Directory for generated waveform files")
    parser.add_argument("--output_prefix", type=str, default="tx_waveform", help="Output file prefix")
    return parser


def load_symbols(points_csv, pair=None, max_symbols=None):
    """Load complex symbols from points CSV and optionally filter by pair and count."""
    symbols = []
    with open(points_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_pair = int(row["pair"])
            if pair is not None and row_pair != pair:
                continue
            i_val = float(row["I"])
            q_val = float(row["Q"])
            symbols.append(i_val + 1j * q_val)

    if len(symbols) == 0:
        raise ValueError("No symbols found. Check --points_csv and --pair.")

    symbols = np.asarray(symbols, dtype=np.complex64)
    if max_symbols is not None:
        symbols = symbols[:max_symbols]
    return symbols


def rrc_taps(beta, sps, span):
    """Generate unit-energy Root Raised Cosine (RRC) filter taps."""
    if not 0.0 <= beta <= 1.0:
        raise ValueError("rrc_beta must be in [0, 1]")
    if sps <= 0 or span <= 0:
        raise ValueError("sps and rrc_span must be positive")

    num_taps = 2 * span * sps + 1
    t = np.arange(-span * sps, span * sps + 1, dtype=np.float64) / float(sps)
    h = np.zeros_like(t)

    for idx, ti in enumerate(t):
        if np.isclose(ti, 0.0):
            h[idx] = 1.0 - beta + 4.0 * beta / np.pi
        elif beta > 0 and np.isclose(abs(ti), 1.0 / (4.0 * beta)):
            h[idx] = (beta / np.sqrt(2.0)) * (
                ((1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta)))
                + ((1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta)))
            )
        else:
            numerator = (
                np.sin(np.pi * ti * (1.0 - beta))
                + 4.0 * beta * ti * np.cos(np.pi * ti * (1.0 + beta))
            )
            denominator = np.pi * ti * (1.0 - (4.0 * beta * ti) ** 2)
            h[idx] = numerator / denominator

    h = h / np.sqrt(np.sum(h**2))
    if num_taps != h.shape[0]:
        raise RuntimeError("Unexpected RRC tap count")
    return h.astype(np.float32)


def upsample(symbols, sps):
    """Insert sps-1 zeros between symbols."""
    up = np.zeros(len(symbols) * sps, dtype=np.complex64)
    up[::sps] = symbols
    return up


def pulse_shape(symbols, sps, beta, span):
    """Apply RRC pulse shaping to complex symbols."""
    taps = rrc_taps(beta=beta, sps=sps, span=span)
    up = upsample(symbols, sps=sps)
    shaped = np.convolve(up, taps.astype(np.complex64), mode="same")
    return shaped.astype(np.complex64), taps


def normalize_power(waveform, target_power=1.0):
    """Scale waveform to target average power E[|x|^2]."""
    avg_power = float(np.mean(np.abs(waveform) ** 2))
    if avg_power <= 0:
        raise ValueError("Waveform has non-positive average power")
    scale = np.sqrt(target_power / avg_power)
    return (waveform * scale).astype(np.complex64), avg_power, float(target_power)


def clip_magnitude(waveform, clip_mag):
    """Clip complex magnitude while preserving phase."""
    if clip_mag is None:
        return waveform
    mag = np.abs(waveform)
    phase = np.angle(waveform)
    mag = np.minimum(mag, clip_mag)
    return (mag * np.exp(1j * phase)).astype(np.complex64)


def apply_frequency_offset(waveform, sample_rate, freq_offset_hz):
    """Apply digital complex mixing e^(j2pi f n / fs)."""
    if np.isclose(freq_offset_hz, 0.0):
        return waveform
    n = np.arange(waveform.shape[0], dtype=np.float64)
    mixer = np.exp(1j * 2.0 * np.pi * freq_offset_hz * n / sample_rate)
    return (waveform * mixer).astype(np.complex64)


def save_waveform_files(waveform, output_dir, output_prefix):
    """Save complex waveform as .npy and interleaved float32 .bin files."""
    os.makedirs(output_dir, exist_ok=True)

    npy_path = os.path.join(output_dir, f"{output_prefix}.npy")
    np.save(npy_path, waveform.astype(np.complex64))

    bin_path = os.path.join(output_dir, f"{output_prefix}_iq_f32.bin")
    interleaved = np.empty(2 * waveform.shape[0], dtype=np.float32)
    interleaved[0::2] = np.real(waveform).astype(np.float32)
    interleaved[1::2] = np.imag(waveform).astype(np.float32)
    interleaved.tofile(bin_path)

    return npy_path, bin_path


def main():
    """Load constellation points, pulse-shape, and export transmit-ready baseband files."""
    args = get_parser().parse_args()

    symbols = load_symbols(args.points_csv, pair=args.pair, max_symbols=args.max_symbols)
    waveform, taps = pulse_shape(symbols, sps=args.sps, beta=args.rrc_beta, span=args.rrc_span)
    waveform, before_power, _ = normalize_power(waveform, target_power=args.target_power)
    waveform = clip_magnitude(waveform, args.clip_mag)
    waveform = (waveform * np.float32(args.gain)).astype(np.complex64)
    waveform = apply_frequency_offset(waveform, sample_rate=args.sample_rate, freq_offset_hz=args.freq_offset_hz)

    npy_path, bin_path = save_waveform_files(waveform, args.output_dir, args.output_prefix)

    metadata = {
        "points_csv": args.points_csv,
        "pair": args.pair,
        "num_symbols": int(symbols.shape[0]),
        "num_samples": int(waveform.shape[0]),
        "sps": int(args.sps),
        "rrc_beta": float(args.rrc_beta),
        "rrc_span": int(args.rrc_span),
        "target_power": float(args.target_power),
        "power_before_norm": float(before_power),
        "power_after_all": float(np.mean(np.abs(waveform) ** 2)),
        "clip_mag": None if args.clip_mag is None else float(args.clip_mag),
        "gain": float(args.gain),
        "sample_rate": float(args.sample_rate),
        "freq_offset_hz": float(args.freq_offset_hz),
        "rrc_tap_count": int(taps.shape[0]),
        "npy_path": npy_path,
        "iq_f32_bin_path": bin_path,
    }

    metadata_path = os.path.join(args.output_dir, f"{args.output_prefix}_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("Saved:", npy_path)
    print("Saved:", bin_path)
    print("Saved:", metadata_path)


if __name__ == "__main__":
    main()
