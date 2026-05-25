This repo is a work-in-progress attempt to construct an encoder-decoder model to predict power spectra.

## Installation

Assuming your environment is "cmb-ml"

- `conda env update -n cmb-ml -f env.yaml`


## For use with cuHPX

Installing everything for use with cuhpx (mostly for the server, newer GPU install):

```
conda create -n cmb-ml-cuhpx python=3.10 -y
conda activate cmb-ml-cuhpx
conda install -c conda-forge cfitsio gsl fftw
# Imports were suggested for pymaster only... maybe the next line would be better.
# CFLAGS="-I${CONDA_PREFIX}/include" LDFLAGS="-L${CONDA_PREFIX}/lib" pip install pymaster

# But this is probably ok. It could mess with other build later, but unlikely
export CFLAGS="-I${CONDA_PREFIX}/include"
export LDFLAGS="-L${CONDA_PREFIX}/lib"
pip install numpy astropy pyyaml pyerfa packaging scikit-build-core cmake ninja pybind11 healpy matplotlib camb python-docx tqdm hydra-core pandas scikit-image nvidia-ml-py3 pymaster
# Torch for newer GPU; older (workstation GPU) uses 2.5.1/cu121
pip install torch==2.6.0 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu124
mkdir -p /bigdata/jim/local/cuda
wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run
sh cuda_12.4.1_550.54.15_linux.run --silent --toolkit --toolkitpath=/bigdata/jim/local/cuda/cuda-12.4
export CUDA_HOME=/bigdata/jim/local/cuda/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
cd /bigdata/cmb_project/code
git clone https://github.com/NVIDIA/torch-harmonics.git
git clone https://github.com/NVlabs/cuHPX.git
cd /bigdata/cmb_project/code/torch-harmonics
pip install --no-build-isolation --no-deps .
cd /bigdata/cmb_project/code/cuHPX
pip install --no-build-isolation --no-deps .
cd /bigdata/cmb_project/code/pysm
pip install .
cd /bigdata/cmb_project/code/cmb-ml
pip install -e .
```
