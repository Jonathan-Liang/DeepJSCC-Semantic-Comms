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
        for pair in range(num_pairs):
            pair_points = symbols[:, 2 * pair: 2 * pair + 2]
            pair_points = maybe_subsample(pair_points, args.max_points, rng)
            out_path = os.path.join(args.output_dir, "{}_pair_{}.png".format(args.output_prefix, pair))
            save_constellation(pair_points, out_path, "Encoder Constellation Pair {}".format(pair))
            print("Saved:", out_path)
    else:
        if args.pair_index >= num_pairs:
            raise ValueError("pair_index={} out of range. Valid range: [0, {}]".format(args.pair_index, num_pairs - 1))
        pair_points = symbols[:, 2 * args.pair_index: 2 * args.pair_index + 2]
        pair_points = maybe_subsample(pair_points, args.max_points, rng)
        out_path = os.path.join(args.output_dir, "{}_pair_{}.png".format(args.output_prefix, args.pair_index))
        save_constellation(pair_points, out_path, "Encoder Constellation Pair {}".format(args.pair_index))
        print("Saved:", out_path)


if __name__ == "__main__":
    main()    main()