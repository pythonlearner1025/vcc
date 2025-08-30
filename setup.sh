pip install numpy pandas scanpy[skmisc] gcsfs pyarrow h5py tqdm matplotlib seaborn wandb nvitop cell-eval xformers rotary-embedding-torch
git clone https://github.com/KellerJordan/Muon
pip install git+https://github.com/KellerJordan/Muon

# Detect GPU type and install appropriate flash_attn wheel
if nvidia-smi | grep -q "A100\|L40"; then
    pip install flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
elif nvidia-smi | grep -q "H100"; then
    pip install flash_attn-2.8.2+cu12torch2.4cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
else
    echo "Warning: Unknown GPU type, defaulting to H100 flash_attn wheel"
    pip install flash_attn-2.8.2+cu12torch2.4cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
fi

cp data/competition_support/competition_train.h5 /
cp esm_all.pt /
cp assets/token_distribution.json / 