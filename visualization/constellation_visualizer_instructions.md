# NOTE
We are using the ISEC implementation of DeepJSCC, because the documentation for it is significantly better than the original implementation. This is because the original implementation is the result of replicating the research, whereas the ISEC is an official implementation of there work. As ISEC is a direct upgrade to DeepJSCC, this should positively impact our research.

# 1) Activate the project venv

Command: source .venv/bin/activate
What it does: switches your shell to use Python/packages from the project virtual environment.
How to tell it worked: your prompt usually shows (.venv).

# 2) Run evaluation (eval.py)

Example:
python eval.py -jmp cifar_16.pb -bmp cifar_16_bfcnn.pb -dset cifar -nc 16 --num_conv_blocks=2 --num_residual_blocks=2 -ne 1 -mb 1 -ni 5 -pf 1
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
python plot_encoder_constellation.py -jmp cifar_16.pb -dset cifar -nc 16 --num_conv_blocks=2 --num_residual_blocks=2 --num_batches 1 --plot_all_pairs --output_prefix cifar_with_table_all --with_table

### Key flags
--num_batches: how much test data to sample
--plot_all_pairs or --pair_index 0
--with_table: save side-by-side plot + table image
--save_table_csv: save summary table CSV
--save_points_csv: save one row per point CSV

# 4) Where outputs go

Default folder: constellation
Prefix controls file names via --output_prefix ....