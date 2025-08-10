from typing import Any, Dict, List, Tuple

import torch

from dataset.scrna_hvg_dataset import create_scrna_hvg_dataloader
from dataset.orion_paired import create_orion_train_val_dataloaders
from dataset.vcc_paired_dataloader import create_vcc_train_val_dataloaders
from tokenizer import create_logbin_tokenizer, TokenizedScRNADataset


def _read_hvgs(hvg_info_path: str) -> List[str]:
    with open(hvg_info_path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def create_orion_finetune(
    *,
    data_dir: str,
    hvg_info_path: str,
    set_size: int = 128,
    vocab_size: int = 128,
    max_value: float = 10.82,
    train_split: float = 0.8,
    batch_size: int = 1,
    num_workers: int = 4,
    random_seed: int = 42,
) -> Tuple[Tuple[Any, torch.utils.data.DataLoader], Tuple[Any, torch.utils.data.DataLoader]]:
    hvg_genes = _read_hvgs(hvg_info_path)
    tokenizer, _ = create_logbin_tokenizer(vocab_size, max_value=max_value)
    (orion_train_ds, orion_train_dl), (orion_val_ds, orion_val_dl) = create_orion_train_val_dataloaders(
        batches_dir=data_dir,
        hvg_gene_ids=hvg_genes,
        tokenizer=tokenizer,
        batch_size=batch_size,
        set_size=set_size,
        train_split=0.9,
        num_workers=4,
        random_seed=42
    )
    # Trainer expects a single dataloader; for finetune we use the train one.
    # If validation is needed per-epoch, expose via hooks in the future.
    return (orion_train_ds, orion_train_dl)

def create_scrna_pretrain(
    *,
    data_dir: str,
    hvg_info_path: str,
    batch_size: int = 128,
    vocab_size: int = 128,
    max_value: float = 10.82,
    num_workers: int = 0,
    use_cache: bool = True,
    normalize: bool = False,
    preload_cache: bool = True,
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
    # Pre-warm cache in parent so DataLoader workers inherit via COW
    if preload_cache and use_cache:
        for i in range(len(scrna_dataset.batch_files)):
            _ = scrna_dataset._load_batch(i)
    tokenizer, _ = create_logbin_tokenizer(vocab_size=vocab_size, max_value=max_value)
    tokenized = TokenizedScRNADataset(scrna_dataset, tokenizer)
    dataloader = torch.utils.data.DataLoader(
        tokenized,
        batch_size=batch_size,
        shuffle=True,
        num_workers=max(1, num_workers),
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=True,
    )
    return tokenized, dataloader

def create_vcc_finetune(
    *,
    adata_path: str,
    hvg_info_path: str,
    set_size: int = 128,
    batch_size: int = 1,
    vocab_size: int = 128,
    max_value: float = 10.82,
    n_samples_per_gene_train: int = 10,
    n_samples_per_gene_val: int = 1,
    train_split: float = 0.8,
    num_workers: int = 4,
    random_seed: int = 42,
    normalize: bool = False,
    blacklist_path: str = "assets/blacklist.txt",
) -> Tuple[Tuple[Any, torch.utils.data.DataLoader], Tuple[Any, torch.utils.data.DataLoader]]:
    hvg_genes = _read_hvgs(hvg_info_path)
    tokenizer, _ = create_logbin_tokenizer(vocab_size, max_value=max_value)
    (vcc_dataset, vcc_dataloader), (val_dataset, val_dataloader) = create_vcc_train_val_dataloaders(
        tokenizer=tokenizer,
        adata_path=adata_path,
        hvg_gene_ids=hvg_genes,
        set_size=set_size,
        n_samples_per_gene_train=n_samples_per_gene_train,
        n_samples_per_gene_val=n_samples_per_gene_val,
        #train_split=train_split,
        batch_size=batch_size,
        num_workers=num_workers,
        random_seed=random_seed,
        normalize=normalize,
        blacklist_path=blacklist_path,
    )
    # Expose validation handles on the returned training dataloader so the generic
    # trainer can pick them up for per-epoch validation without special-casing.
    try:
        setattr(vcc_dataloader, "val_dataloader", val_dataloader)
        setattr(vcc_dataloader, "val_dataset", val_dataset)
    except Exception:
        pass
    # Trainer expects a single dataloader; for finetune we use the train one.
    # Validation is accessed via attributes set above.
    return (vcc_dataset, vcc_dataloader)