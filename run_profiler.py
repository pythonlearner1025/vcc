# scripts/profile_train_epoch.py
import torch
from torch.profiler import profile, ProfilerActivity           # ⬅ profiling
from train_orion import train_epoch_st, ConditionalModelConfig        # your code
from models.diffusion import AbsorbingMaskMD4Continuous, get_lr, create_muon_optimizer            # your code
from pathlib import Path

# -------------------------------------------------------------------------
# 1.  Build a tiny dataloader & objects that `train_epoch_st` expects
#    (replace this with the real ones you already use)
# -------------------------------------------------------------------------
config        = ConditionalModelConfig()                # ← your config class
dummy_tokens = torch.randint(0, config.vocab_size, (8, 6822))          # 8 cells, 256 tokens
loader        = [dummy_tokens] * 12                     # > 11 steps available
from models.diffusion import ConditionalDiffusionTransformer

model         = ConditionalDiffusionTransformer(config).cuda()
diffusion     = AbsorbingMaskMD4Continuous(config)         # ← your impl
opt           = create_muon_optimizer(model, config)
sch           = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: get_lr(opt, it, config)) 

# -------------------------------------------------------------------------
# 2.  Wrap the first 11 steps (1 warm-up + 10 active) with a profiler
# -------------------------------------------------------------------------
activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

with profile(activities=activities,
             profile_memory=True,                 # track CUDA memory
             record_shapes=False,
             schedule=torch.profiler.schedule(    # 1 warm-up / 10 active
                 wait=0, warmup=1, active=10, repeat=1)
             ) as prof:

    global_step = 0
    # **important** – tell the profiler when a new step begins
    for step, batch in enumerate(loader):
        global_step = train_epoch_st(
            model, diffusion, loader, (opt, sch), config,
            epoch=0, global_step=global_step, total_training_steps=10,
            checkpoint_dir=Path('.'),
            tokenizer=None, batch_to_idx=None,
            use_control_sets=False, max_cells=None
        )
        prof.step()                                        # ← profiler tick
        if step >= 10: break                               # done after 11

# -------------------------------------------------------------------------
# 3.  Pretty-print CUDA kernel time  &  CUDA memory usage
# -------------------------------------------------------------------------
print("\n=====   CUDA TIME (10 steps)   =====")
print(prof.key_averages().table(
      sort_by="self_cuda_time_total", row_limit=15))

print("\n=====   CUDA MEMORY (10 steps) =====")
print(prof.key_averages().table(
      sort_by="self_cuda_memory_usage", row_limit=15))