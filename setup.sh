pip install numpy pandas scanpy[skmisc] gcsfs pyarrow h5py tqdm torch matplotlib seaborn wandb nvitop

# Only install flash attention if torch 2.4 is installed
python3 -c "import torch; v=torch.__version__; exit(0 if not v.startswith('2.4') else 1)" || \
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.2/flash_attn-2.8.2%2Bcu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# copy data to root dir for faster loading
cp -r data_normalized /
cp esm_all.pt /
cp token_distribution.json / 
cp -r data/competition_support/competition_train.h5 /
