### Generic YAML-driven Trainer

Run training with a single command, defining all behavior in a YAML config. No code edits required for new experiments.

Example:

```
python -u train.py --config configs/example_scrna_vcc.yml
```

#### Config structure
- `model.model_class`: Dotted path to a model class. It will be constructed with `model_args` and moved to the configured device.
- `optimizer.optimizer_class`: Dotted path to an optimizer class or factory; receives the model parameters and `optimizer_args`.
- `optimizer.scheduler_class` (optional): Dotted path to a scheduler class; receives `scheduler_args`.
- `datasets`: Ordered list of dataset phases. Each phase defines:
  - `dataloader_factory`: Dotted path to a function returning `(dataset, dataloader)` or just a `DataLoader`.
  - `dataloader_args`: Keyword arguments forwarded to the factory.
  - `epochs`, `log_every_steps`, `save_every_steps`, `eval_every_epochs`, `max_steps_per_epoch`, `lr`.

#### Dataloader compliance
- The dataloader must yield either:
  - a `torch.Tensor` of shape `[B, N]` named tokens (pretrain style), or
  - a `dict` with keys `tokens` and optional `control`, `target_gene_idx`, `batch_idx`.

#### Global technical batch indexing
- The trainer discovers local batch names per phase from:
  - `dataset.unique_batches` (if provided), or
  - `dataset.adata.obs['batch']` (if present), or
  - falls back to a single per-phase batch name.
- It merges all names into a unique global mapping and:
  - sets `model.config.n_technical_batches` accordingly (rounded up to multiple of 16 when FP8 is enabled),
  - attaches `batch_to_idx` to datasets/collators when supported,
  - wraps the collate function so each batch has a `batch_idx` tensor if missing.

The trainer will adapt batches automatically and call a loss via one of:
- `model.diffusion.compute_loss(model, tokens, control_set=..., target_gene_idx=..., batch_idx=..., step=...)`
- `model.compute_loss(batch)`
- `model.forward(batch)` + `model.criterion(logits, batch)`

#### Checkpointing
- Checkpoints are saved to `checkpoints/<run_name>_<timestamp>/checkpoint_*.pt` and include model, optimizer, scheduler states, and global counters.
- To resume, set `resume_from: /path/to/checkpoint.pt` in the config.