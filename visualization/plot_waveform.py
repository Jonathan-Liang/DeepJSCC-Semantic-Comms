import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def get_parser():
    """Create parser for waveform visualization options."""
    parser = argparse.ArgumentParser(description="Plot complex baseband waveform in time and frequency domains.")
    parser.add_argument("--input_npy", type=str, default=None, help="Path to complex64 .npy waveform")
    parser.add_argument("--input_iq_f32", type=str, default=None, help="Path to interleaved float32 IQ .bin file")
    parser.add_argument("--sample_rate", type=float, default=1.0, help="Sample rate in Hz")
    parser.add_argument("--num_time_samples", type=int, default=2000, help="Number of time samples to plot")
    parser.add_argument("--fft_size", type=int, default=8192, help="FFT size for spectrum plot")
    parser.add_argument("--save_time", action="store_true", help="Also save time-domain I/Q plot")
    parser.add_argument("--save_iq", action="store_true", help="Also save IQ scatter plot")
    parser.add_argument("--output_dir", type=str, default="./outputs/constellation", help="Output directory")
    parser.add_argument("--output_prefix", type=str, default="waveform", help="Output image prefix")
    return parser


def load_waveform(input_npy=None, input_iq_f32=None):
    """Load complex waveform from npy or interleaved IQ binary file."""
    if input_npy is not None:
        waveform = np.load(input_npy)
        return waveform.astype(np.complex64)

    if input_iq_f32 is not None:
        iq = np.fromfile(input_iq_f32, dtype=np.float32)
        if iq.size % 2 != 0:
            raise ValueError("IQ binary size must be even (I/Q interleaved)")
        waveform = iq[0::2] + 1j * iq[1::2]
        return waveform.astype(np.complex64)

    raise ValueError("Provide --input_npy or --input_iq_f32")


def save_time_plot(waveform, sample_rate, num_time_samples, out_path):
    """Save I and Q versus time plot."""
    n = min(num_time_samples, waveform.shape[0])
    t = np.arange(n) / float(sample_rate)
    i_vals = np.real(waveform[:n])
    q_vals = np.imag(waveform[:n])

    plt.figure(figsize=(10, 4))
    plt.plot(t, i_vals, linewidth=1.0, label="I")
    plt.plot(t, q_vals, linewidth=1.0, label="Q")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Waveform (Time Domain)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_spectrum_plot(waveform, sample_rate, fft_size, out_path):
    """Save centered power spectrum plot in dB."""
    n = min(max(256, fft_size), waveform.shape[0])
    x = waveform[:n]
    window = np.hanning(n)
    xw = x * window
    spectrum = np.fft.fftshift(np.fft.fft(xw, n=n))
    power = np.abs(spectrum) ** 2
    power_db = 10.0 * np.log10(power + 1e-12)

    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / sample_rate))
    freqs_mhz = freqs / 1e6

    plt.figure(figsize=(10, 4))
    plt.plot(freqs_mhz, power_db, linewidth=1.0)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Power (dB)")
    plt.title("Waveform Spectrum")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_iq_plot(waveform, num_time_samples, out_path):
    """Save IQ trajectory/scatter plot for a subset of samples."""
    n = min(num_time_samples, waveform.shape[0])
    i_vals = np.real(waveform[:n])
    q_vals = np.imag(waveform[:n])

    plt.figure(figsize=(6, 6))
    plt.scatter(i_vals, q_vals, s=3, alpha=0.35)
    plt.axhline(0.0, color="gray", linewidth=0.8)
    plt.axvline(0.0, color="gray", linewidth=0.8)
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.title("Waveform IQ Samples")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    """Load waveform and save time/spectrum/IQ visualizations."""
    args = get_parser().parse_args()
    waveform = load_waveform(input_npy=args.input_npy, input_iq_f32=args.input_iq_f32)

    os.makedirs(args.output_dir, exist_ok=True)
    spec_path = os.path.join(args.output_dir, f"{args.output_prefix}_spectrum.png")
    save_spectrum_plot(waveform, sample_rate=args.sample_rate, fft_size=args.fft_size, out_path=spec_path)

    print("Saved:", spec_path)

    if args.save_time:
        time_path = os.path.join(args.output_dir, f"{args.output_prefix}_time.png")
        save_time_plot(waveform, sample_rate=args.sample_rate, num_time_samples=args.num_time_samples, out_path=time_path)
        print("Saved:", time_path)

    if args.save_iq:
        iq_path = os.path.join(args.output_dir, f"{args.output_prefix}_iq.png")
        save_iq_plot(waveform, num_time_samples=args.num_time_samples, out_path=iq_path)
        print("Saved:", iq_path)


if __name__ == "__main__":
    main()
