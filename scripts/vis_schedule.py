import numpy as np
import math
import matplotlib.pyplot as plt

# Parameters
n_timesteps = 32
s = 0.008
finetune_mask_ratio_end = 0.8
pretrain_mask_ratio = 1.0

# Steps
steps = np.arange(n_timesteps + 1, dtype=np.float32)
alphas = np.cos((steps / n_timesteps + s) / (1 + s) * math.pi * 0.5) ** 2
alphas = alphas / alphas[0]
base_probs = 1 - alphas[:-1]

# Avoid division by zero
eps = 1e-8
p_mask_probs = base_probs * (finetune_mask_ratio_end / (base_probs[-1] + eps))
p_mask_probs_pretrain = base_probs * (pretrain_mask_ratio / (base_probs[-1] + eps))

# Visualization
plt.figure(figsize=(8, 5))
plt.plot(range(n_timesteps), p_mask_probs, marker='o', label='Finetune Mask Probabilities')
plt.plot(range(n_timesteps), p_mask_probs_pretrain, marker='x', label='Pretrain Mask Probabilities', linestyle='--')
plt.xlabel("Timestep")
plt.ylabel("Mask Probability")
plt.title("Masking Schedule (n_timesteps=16, mask_ratio_end=1.0)")
plt.legend()
plt.grid(True)
plt.savefig('masking_schedule.png')
plt.close()
