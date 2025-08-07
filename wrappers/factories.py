from typing import Any, Dict, List, Tuple

import torch

from dataset.scrna_hvg_dataset import create_scrna_hvg_dataloader
from dataset.vcc_paired_dataloader import create_vcc_train_val_dataloaders
from tokenizer import create_logbin_tokenizer, create_delta_tokenizer, TokenizedScRNADataset


def _read_hvgs(hvg_info_path: str) -> List[str]:
    with open(hvg_info_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def create_scrna_pretrain(
    *,
    data_dir: str,
    hvg_info_path: str,
    batch_size: int = 128,
    num_workers: int = 0,
    use_cache: bool = True,
    normalize: bool = False,
) -> Tuple[Any, torch.utils.data.DataLoader]:
    hvg_genes = _read_hvgs(hvg_info_path)
    scrna_dataset, _ = create_scrna_hvg_dataloader(
        data_dir=data_dir,
        hvg_genes=hvg_genes,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        use_cache=use_cache,
        normalize=normalize,
    )
    tokenizer, _ = create_logbin_tokenizer(vocab_size=128, max_value=9.2)
    tokenized = TokenizedScRNADataset(scrna_dataset, tokenizer)
    dataloader = torch.utils.data.DataLoader(
        tokenized,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return tokenized, dataloader


def create_vcc_finetune(
    *,
    adata_path: str,
    hvg_info_path: str,
    set_size: int = 128,
    batch_size: int = 1,
    n_samples_per_gene_train: int = 10,
    n_samples_per_gene_val: int = 1,
    train_split: float = 0.8,
    num_workers: int = 4,
    random_seed: int = 42,
    normalize: bool = False,
    blacklist_path: str = "assets/blacklist.txt",
) -> Tuple[Tuple[Any, torch.utils.data.DataLoader], Tuple[Any, torch.utils.data.DataLoader]]:
    hvg_genes = _read_hvgs(hvg_info_path)
    tokenizer, _ = create_delta_tokenizer(vocab_size=128, max_abs=9.2, min_abs=1e-3)
    (vcc_dataset, vcc_dataloader), (val_dataset, val_dataloader) = create_vcc_train_val_dataloaders(
        tokenizer=tokenizer,
        adata_path=adata_path,
        hvg_gene_ids=hvg_genes,
        set_size=set_size,
        batch_size=batch_size,
        n_samples_per_gene_train=n_samples_per_gene_train,
        n_samples_per_gene_val=n_samples_per_gene_val,
        train_split=train_split,
        num_workers=num_workers,
        random_seed=random_seed,
        normalize=normalize,
        blacklist_path=blacklist_path,
    )
    # Trainer expects a single dataloader; for finetune we use the train one.
    # If validation is needed per-epoch, expose via hooks in the future.
    return (vcc_dataset, vcc_dataloader)