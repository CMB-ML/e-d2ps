# CMB-ML: D2PS

This repository implements Direct-to-Power-Spectrum (D2PS), a machine learning method for predicting theoretical CMB temperature angular power spectra directly from multi-frequency observation maps.

This repository relies on the [CMB-ML core repository](https://github.com/CMB-ML/cmb-ml) for simulation generation, data management, and pipeline execution. It also contains the comparative analysis pipeline used to prepare power spectra from D2PS, RealizationPS, CNILC, and cmbNNCS, reproducing the results of the D2PS paper.

## Installation

Follow the installation instructions in the [CMB-ML repository](https://github.com/CMB-ML/cmb-ml), using release `v0.3.0-d2ps`.

With the CMB-ML conda environment activated, and from the CMB-ML repository directory, clone this repository alongside it and install its additional dependencies:
```bash
    cd ..
    git clone https://github.com/CMB-ML/e-d2ps.git
    cd e-d2ps
    git checkout v0.2.0-d2ps
    conda env update -n cmb-ml -f env.yaml
    python -m pip check
```

A compatible NVIDIA GPU and driver are required for GPU execution. The provided environment specifies PyTorch 2.6.0 with CUDA 12.4.

We have seen small numerical differences in predictions from the same trained model when using different hardware and software environments. Reproduction of the reported results was verified using an NVIDIA V100.

## Dataset

To reproduce the results reported in the D2PS paper, use the `I_512_1850_UnifWMAP_Wide` dataset, which can be obtained with the dataset download instructions in the CMB-ML repository.

Ensure that the `CMB_ML_DATA` environment variable is set as described in the CMB-ML documentation.

The configurations for training, inference, and comparative analysis are set up to use this dataset by default.

## Released Model Assets

The trained model used in the D2PS paper and its supporting assets are available from [Box](https://utdallas.box.com/s/eiqlxcyfq564fdshbmiqerl0q5ftgrm5).

The set of files is:

| File | Size | Description |
|---|---:|---|
| `model_best_ep102.pth` | 13.88 GiB | Trained model checkpoint, including optimizer state |
| `obs2ps_SimpleEncDec_polynomial18_init_1.pth` | 4.63 GiB | Model initialization checkpoint used for training |
| `D2PS_r22_supporting_assets.tar` | 566.4 MiB | Preprocessing assets, training records, and supporting data |
| `D2PS_r22_SHA256SUMS.txt` | 294 bytes | SHA-256 checksum manifest |

Download the files using `rclone`:

```bash
mkdir -p D2PS_assets
cd D2PS_assets

BOX_URL='https://utdallas.app.box.com/index.php?rm=box_download_shared_file&shared_name=eiqlxcyfq564fdshbmiqerl0q5ftgrm5'

rclone copyurl \
    "${BOX_URL}&file_id=f_2593809978379" \
    ./model_best_ep102.pth --progress

rclone copyurl \
    "${BOX_URL}&file_id=f_2593806443707" \
    ./obs2ps_SimpleEncDec_polynomial18_init_1.pth --progress

rclone copyurl \
    "${BOX_URL}&file_id=f_2593802629358" \
    ./D2PS_r22_supporting_assets.tar --progress

cd ..
```

The data has the following properties:

| Filename | Size (bytes) | SHA-256 |
|---|---:|---|
| `model_best_ep102.pth` | 14900286194 | `2f018b17afc6bfc27c8514dfca6033d35e44efa7cc7192f688f317d044edf18c` |
| `obs2ps_SimpleEncDec_polynomial18_init_1.pth` | 4966530614 | `a1e37de6c33b46c5d74d1c4b62c405a703021f52a32a92d1568d36c5e5e91648` |
| `D2PS_r22_supporting_assets.tar` | 593909760 | `4a54d334a4f6c9ef972d388906eb538a5c98089ed4d3a4d16389b8ef1e442179` |

Automatic verification of the checksum can be performed with:
```bash
cd D2PS_assets
rclone copyurl \
    "${BOX_URL}&file_id=f_2593799241089" \
    ./D2PS_r22_SHA256SUMS.txt

sha256sum -c D2PS_r22_SHA256SUMS.txt
cd ..
```

After verifying the checksums, place the assets in the dataset directory using the following commands:

```bash
DATASET_DIR="$CMB_ML_DATA/Datasets/I_512_1850_UnifWMAP_Wide"

MODEL_REL="obs2ps_r22/drop_no_drop_scale_ps_None-scale_px_minmax_global-optimizer_Adam-lr_0.001-norm_bn-activation_gelu-down_XXXXXXXXX-spat_max"

MODEL_DIR="$DATASET_DIR/$MODEL_REL/F_Model"
INIT_DIR="$DATASET_DIR/d2ps-prep/F_Model_Init"

mkdir -p "$MODEL_DIR" "$INIT_DIR"

tar -xf D2PS_assets/D2PS_r22_supporting_assets.tar -C "$DATASET_DIR"

cp D2PS_assets/model_best_ep102.pth "$MODEL_DIR/"
cp D2PS_assets/obs2ps_SimpleEncDec_polynomial18_init_1.pth "$INIT_DIR/"
```

## Running D2PS

The entry point for preprocessing, training, and inference is `main_d2ps.py`.

### Reproducing the Reported Results

The provided model allows reproduction of the reported results without retraining. To do so, in `main_d2ps.py`, modify the execution block at the bottom of the file as follows:

```python
if __name__ == "__main__":
    # run_1_prelim_px()
    # run_2_prelim_ps()
    run_3_prep_px()
    # run_4_prep_ps()
    # run_5_make_models()
    # run_6_train()
    run_7_infer()
```

Run the pipeline:

```bash
python main_d2ps.py
```

The default configuration uses the provided `best` checkpoint and generates predicted power spectra for the validation and test splits (`Valid`, `Test`, `TestFP`, and `TestFFN`).

For a smaller test run, the number of simulations processed during inference can be limited:

```bash
python main_d2ps.py 'pipeline.preprocess_maps.splits=[test,valid]' n_infer_cap=12
```

The resulting power spectra are stored in the `G_Predict/` subdirectory of the model working directory specified above.

### Training

To train a new model and generate its predictions, run the complete pipeline using `main_d2ps.py` with its original execution block intact.

```python
if __name__ == "__main__":
    run_1_prelim_px()
    # run_2_prelim_ps()
    run_3_prep_px()
    run_4_prep_ps()
    run_5_make_models()
    run_6_train()
    run_7_infer()
```

To preserve the released preprocessing assets, provided model, and associated predictions, modify the following settings in `cfg/config_d2ps.yaml` to use new working directories:

```yaml
working_dir: &wd "D2PS_New/"
prep_working_dir: "d2ps-prep-new/"
```

The pipeline can be run with:

```bash
python main_d2ps.py
```

The default configuration uses the SimpleEncDec architecture, a batch size of 4, and 300 training epochs.

Training parameters can be modified in the configuration files or overridden from the command line using Hydra.

### Comparative Analysis

The entry point for comparative analysis is `main_comp.py`. This pipeline prepares the power spectra from D2PS, RealizationPS, CNILC, and cmbNNCS for comparison against their corresponding theoretical power spectra.

Before running the comparative analysis pipeline, generate predictions and power spectra using the instructions provided in the respective repositories:

- **D2PS:** Follow the inference instructions above.
- **CNILC:** Follow the instructions in the [CMB-ML PyILC repository](https://github.com/CMB-ML/cmb-ml-pyilc).
- **cmbNNCS:** Follow the instructions in the [CMB-ML cmbNNCS repository](https://github.com/CMB-ML/cmb-ml-cmbnncs).

The RealizationPS power spectra are generated as part of the comparative analysis pipeline.

Run the pipeline:

```bash
python main_comp.py
```

The pipeline collects the predicted and theoretical power spectra, selects polynomial fitting degrees using the validation set, and applies polynomial smoothing to the RealizationPS, CNILC, and cmbNNCS power spectra. No additional polynomial smoothing is applied to the D2PS predictions.

The resulting power spectra and supporting analysis data are stored in the `Analysis_Bundle/` subdirectories of the respective method working directories within the dataset directory.

## Licensing

- **Source code:** [MIT License](LICENSE).
- **Released model and supporting data assets:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE.data). A copy of the license is also available alongside the downloadable assets on Box.

## Attribution

If you use this repository in your research, please cite the relevant publications:

- **D2PS:** Citation forthcoming.
- **CMB-ML simulation framework:** [James Amato, Yunan Xie, Leonel Medina-Varela, et al. 2025, ICCV, 9418–9430](https://doi.org/10.1109/ICCV51701.2025.00879).
