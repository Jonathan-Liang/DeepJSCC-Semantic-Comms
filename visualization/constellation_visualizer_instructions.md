# NOTE
We are using the ISEC implementation of DeepJSCC, because the documentation for it is significantly better than the original implementation. This is because the original implementation is the result of replicating the research, whereas the ISEC is an official implementation of there work. As ISEC is a direct upgrade to DeepJSCC, this should positively impact our research.

# 1) Activate the project venv

Command: source .venv/bin/activate
What it does: switches your shell to use Python/packages from the project virtual environment.
How to tell it worked: your prompt usually shows (.venv).

# 2) Run evaluation (eval.py)

Example:
python eval.py -jmp saved_models/cifar_16.pb -bmp saved_models/cifar_16_bfcnn.pb -dset cifar -nc 16 --num_conv_blocks=2 --num_residual_blocks=2 -ne 1 -mb 1 -ni 5 -pf 1
### Key flags
-jmp: JSCC model checkpoint
-bmp: BFCNN checkpoint
-dset: dataset (cifar, openimages, kodak)
-nc: latent channels
-ne: number of experiments
-mb: max test batches
-ni: SEC iterations
-pf: print frequency

# 3) Plot encoder constellation (plot_encoder_constellation.py)

## Example (all pairs + table):
python plot_encoder_constellation.py -jmp saved_models/cifar_16.pb -dset cifar -nc 16 --num_conv_blocks=2 --num_residual_blocks=2 --num_batches 1 --plot_all_pairs --output_prefix cifar_with_table_all --with_table

### Key flags
--num_batches: how much test data to sample
--plot_all_pairs or --pair_index 0
--with_table: save side-by-side plot + table image
--save_table_csv: save summary table CSV
--save_points_csv: save one row per point CSV

# 4) Where outputs go

Default folder: constellation
Prefix controls file names via --output_prefix ....

# 5) Convert constellation points to transmit waveform

First export constellation points from the plotter:
python plot_encoder_constellation.py -jmp saved_models/cifar_16.pb -dset cifar -nc 16 --num_conv_blocks=2 --num_residual_blocks=2 --num_batches 1 --plot_all_pairs --save_points_csv --output_prefix cifar_points

Then convert points to pulse-shaped complex baseband:
python constellation_to_waveform.py --points_csv outputs/constellation/cifar_points_points.csv --pair 0 --sps 8 --rrc_beta 0.25 --rrc_span 10 --sample_rate 1000000 --output_prefix cifar_pair0_tx

Outputs:
- .npy complex64 waveform
- _iq_f32.bin interleaved float32 (I,Q) for SDR tools
- _metadata.json with waveform and filter parameters

# 6) Plot waveform so it is visible

Example:
python plot_waveform.py --input_npy outputs/constellation/cifar_pair0_tx.npy --sample_rate 1000000 --output_prefix cifar_pair0_tx

Generated plots:
- _spectrum.png (frequency-domain power, always generated)

Optional extra plots (only when requested):
- add --save_time for _time.png (I/Q vs time)
- add --save_iq for _iq.png (IQ sample cloud)