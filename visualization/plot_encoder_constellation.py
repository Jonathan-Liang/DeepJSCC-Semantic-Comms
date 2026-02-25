import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config
import loader
import models.autoencoders as ae


def get_parser():
    parser = config.get_common_parser()
    parser.add_argument("--jscc_model_path", "-jmp", type=str, required=True, help="Path to JSCC checkpoint")
    parser.add_argument("--num_batches", type=int, default=1, help="How many test batches to sample")
    parser.add_argument("--max_points", type=int, default=10000, help="Max points per constellation plot")
    parser.add_argument("--pair_index", type=int, default=0, help="Channel pair index for I/Q plot")
    parser.add_argument("--plot_all_pairs", action="store_true", help="Plot all channel pairs (0/1,2/3,...)")
    parser.add_argument("--output_dir", type=str, default="./outputs/constellation", help="Directory to save plots")
    parser.add_argument("--output_prefix", type=str, default="encoder_constellation", help="Output file prefix")
    parser.add_argument("--with_table", action="store_true", help="Save a side-by-side constellation + math table figure")
    parser.add_argument("--save_table_csv", action="store_true", help="Save per-pair math/stat table as CSV")
    parser.add_argument("--save_points_csv", action="store_true", help="Save one row per constellation point to CSV")
    return parser


def build_model(args, dev):
    if "cifar" in args.dataset:
        encoder_cls = ae.Encoder_CIFAR
        decoder_cls = ae.Decoder_CIFAR
    else:
        encoder_cls = ae.Encoder
        decoder_cls = ae.Decoder

    encoder = encoder_cls(
        num_out=args.num_channels,
        num_hidden=args.num_hidden,
        num_conv_blocks=args.num_conv_blocks,
        num_residual_blocks=args.num_residual_blocks,
        normalization=nn.BatchNorm2d,
        activation=nn.PReLU,
        power_norm=args.power_norm,
    )

    decoder = decoder_cls(
        num_in=args.num_channels,
        num_hidden=args.num_hidden,
        num_conv_blocks=args.num_conv_blocks,
        num_residual_blocks=args.num_residual_blocks,
        normalization=nn.BatchNorm2d,
        activation=nn.PReLU,
        no_tanh=False,
    )

    net = ae.Generator(encoder, decoder)
    net.load_state_dict(torch.load(args.jscc_model_path, map_location=dev))
    net.to(torch.device(dev))
    net.eval()
    return net


def sample_encoder_symbols(net, dataloader, device, num_batches):
    symbols = []
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if i >= num_batches:
                break
            inputs = data[0].to(device)
            latent = net.encoder(inputs)
            flattened = latent.permute(0, 2, 3, 1).reshape(-1, latent.size(1)).detach().cpu().numpy()
            symbols.append(flattened)

    if len(symbols) == 0:
        raise RuntimeError("No symbols sampled. Check dataset and num_batches.")
    return np.concatenate(symbols, axis=0)


def maybe_subsample(points, max_points, rng):
    if points.shape[0] <= max_points:
        return points
    indices = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[indices]


def save_constellation(points, out_path, title):
    plt.figure(figsize=(6, 6))
    plt.scatter(points[:, 0], points[:, 1], s=3, alpha=0.2)
    plt.axhline(0.0, color="gray", linewidth=0.8)
    plt.axvline(0.0, color="gray", linewidth=0.8)
    plt.xlabel("In-phase (I)")
    plt.ylabel("Quadrature (Q)")
    plt.title(title)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def pair_stats(points):
    i_vals = points[:, 0]
    q_vals = points[:, 1]
    radius = np.sqrt(i_vals**2 + q_vals**2)
    return {
        "N": int(points.shape[0]),
        "mean(I)": float(np.mean(i_vals)),
        "mean(Q)": float(np.mean(q_vals)),
        "var(I)": float(np.var(i_vals)),
        "var(Q)": float(np.var(q_vals)),
        "cov(I,Q)": float(np.cov(i_vals, q_vals)[0, 1]),
        "mean(|s|)": float(np.mean(radius)),
        "mean(|s|^2)": float(np.mean(radius**2)),
    }


def save_constellation_with_table(points, out_path, pair, title, stats):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [3, 2]})

    ax_scatter = axes[0]
    ax_scatter.scatter(points[:, 0], points[:, 1], s=3, alpha=0.2)
    ax_scatter.axhline(0.0, color="gray", linewidth=0.8)
    ax_scatter.axvline(0.0, color="gray", linewidth=0.8)
    ax_scatter.set_xlabel("In-phase (I)")
    ax_scatter.set_ylabel("Quadrature (Q)")
    ax_scatter.set_title(title)
    ax_scatter.set_aspect("equal", adjustable="box")

    ax_table = axes[1]
    ax_table.axis("off")
    rows = [
        ["Expression", "Definition"],
        ["s_k", "s_k = I_k + jQ_k"],
        ["I_k", "z[:, 2k]"],
        ["Q_k", "z[:, 2k+1]"],
        ["k", "{}".format(pair)],
        ["N", "{}".format(stats["N"])],
        ["mean(I)", "{:.5f}".format(stats["mean(I)"])],
        ["mean(Q)", "{:.5f}".format(stats["mean(Q)"])],
        ["var(I)", "{:.5f}".format(stats["var(I)"])],
        ["var(Q)", "{:.5f}".format(stats["var(Q)"])],
        ["cov(I,Q)", "{:.5f}".format(stats["cov(I,Q)"])],
        ["mean(|s|)", "{:.5f}".format(stats["mean(|s|)"])],
        ["mean(|s|^2)", "{:.5f}".format(stats["mean(|s|^2)"])],
    ]
    table = ax_table.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_stats_csv(csv_path, rows):
    fieldnames = [
        "pair",
        "N",
        "mean(I)",
        "mean(Q)",
        "var(I)",
        "var(Q)",
        "cov(I,Q)",
        "mean(|s|)",
        "mean(|s|^2)",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_points_csv(csv_path, rows):
    fieldnames = ["pair", "point_index", "I", "Q", "abs_s", "phase_rad"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def points_to_rows(points, pair):
    rows = []
    for point_index in range(points.shape[0]):
        i_val = float(points[point_index, 0])
        q_val = float(points[point_index, 1])
        abs_s = float(np.sqrt(i_val**2 + q_val**2))
        phase_rad = float(np.arctan2(q_val, i_val))
        rows.append(
            {
                "pair": int(pair),
                "point_index": point_index,
                "I": i_val,
                "Q": q_val,
                "abs_s": abs_s,
                "phase_rad": phase_rad,
            }
        )
    return rows


def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.num_channels < 2:
        raise ValueError("num_channels must be >= 2 for constellation plotting")
    if args.pair_index < 0:
        raise ValueError("pair_index must be non-negative")

    dev = "cuda:{}".format(args.gpu) if args.gpu >= 0 else "cpu"
    device = torch.device(dev)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    dataloader = loader.get_test_dataloader(args)
    net = build_model(args, dev)
    symbols = sample_encoder_symbols(net, dataloader, device, args.num_batches)

    os.makedirs(args.output_dir, exist_ok=True)
    num_pairs = symbols.shape[1] // 2
    if num_pairs == 0:
        raise ValueError("Encoder output channels must be at least 2")

    if args.plot_all_pairs:
        csv_rows = []
        points_rows = []
        for pair in range(num_pairs):
            pair_points_full = symbols[:, 2 * pair: 2 * pair + 2]
            pair_points = maybe_subsample(pair_points_full, args.max_points, rng)
            stats = pair_stats(pair_points)

            out_path = os.path.join(args.output_dir, "{}_pair_{}.png".format(args.output_prefix, pair))
            save_constellation(pair_points, out_path, "Encoder Constellation Pair {}".format(pair))
            print("Saved:", out_path)

            if args.with_table:
                table_path = os.path.join(args.output_dir, "{}_pair_{}_table.png".format(args.output_prefix, pair))
                save_constellation_with_table(
                    pair_points,
                    table_path,
                    pair,
                    "Encoder Constellation Pair {}".format(pair),
                    stats,
                )
                print("Saved:", table_path)

            csv_rows.append({"pair": pair, **stats})
            if args.save_points_csv:
                points_rows.extend(points_to_rows(pair_points_full, pair))

        if args.save_table_csv:
            csv_path = os.path.join(args.output_dir, "{}_stats.csv".format(args.output_prefix))
            write_stats_csv(csv_path, csv_rows)
            print("Saved:", csv_path)
        if args.save_points_csv:
            points_csv_path = os.path.join(args.output_dir, "{}_points.csv".format(args.output_prefix))
            write_points_csv(points_csv_path, points_rows)
            print("Saved:", points_csv_path)
    else:
        if args.pair_index >= num_pairs:
            raise ValueError("pair_index={} out of range. Valid range: [0, {}]".format(args.pair_index, num_pairs - 1))
        pair_points_full = symbols[:, 2 * args.pair_index: 2 * args.pair_index + 2]
        pair_points = maybe_subsample(pair_points_full, args.max_points, rng)
        stats = pair_stats(pair_points)

        out_path = os.path.join(args.output_dir, "{}_pair_{}.png".format(args.output_prefix, args.pair_index))
        save_constellation(pair_points, out_path, "Encoder Constellation Pair {}".format(args.pair_index))
        print("Saved:", out_path)

        if args.with_table:
            table_path = os.path.join(args.output_dir, "{}_pair_{}_table.png".format(args.output_prefix, args.pair_index))
            save_constellation_with_table(
                pair_points,
                table_path,
                args.pair_index,
                "Encoder Constellation Pair {}".format(args.pair_index),
                stats,
            )
            print("Saved:", table_path)

        if args.save_table_csv:
            csv_path = os.path.join(args.output_dir, "{}_stats.csv".format(args.output_prefix))
            write_stats_csv(csv_path, [{"pair": args.pair_index, **stats}])
            print("Saved:", csv_path)
        if args.save_points_csv:
            points_csv_path = os.path.join(args.output_dir, "{}_points.csv".format(args.output_prefix))
            write_points_csv(points_csv_path, points_to_rows(pair_points_full, args.pair_index))
            print("Saved:", points_csv_path)


if __name__ == "__main__":
    main()