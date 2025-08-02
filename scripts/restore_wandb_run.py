#!/usr/bin/env python3
"""
Script to restore a deleted wandb run from cached data.
"""

import wandb
import yaml
import json
import os
import shutil
from pathlib import Path

def restore_run(run_dir_path):
    """Restore a wandb run from a cached directory."""
    
    run_path = Path(run_dir_path)
    files_path = run_path / "files"
    
    # Read the configuration
    with open(files_path / "config.yaml", 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Extract the actual config values (skip the _wandb metadata)
    config = {}
    for key, value in config_data.items():
        if key != '_wandb' and isinstance(value, dict) and 'value' in value:
            config[key] = value['value']
        elif key != '_wandb':
            config[key] = value
    
    # Read the summary (final metrics)
    with open(files_path / "wandb-summary.json", 'r') as f:
        summary = json.load(f)
    
    # Read metadata for additional info
    with open(files_path / "wandb-metadata.json", 'r') as f:
        metadata = json.load(f)
    
    print(f"Restoring run with config keys: {list(config.keys())}")
    print(f"Final metrics: {summary}")
    
    # Initialize a new wandb run
    run = wandb.init(
        project="vcc-st-diffusion",  # Match original/restored project
        config=config,
        tags=["restored"],
        notes=f"Restored from deleted run at {metadata.get('startedAt', 'unknown time')}"
    )
    
    try:
        # Log the final metrics
        for key, value in summary.items():
            if not key.startswith('_'):  # Skip wandb internal keys
                run.summary[key] = value
        
        # Upload important files as artifacts
        artifact = wandb.Artifact(
            name=f"restored-run-files",
            type="run-files"
        )
        
        # Add the log file if it exists
        output_log = files_path / "output.log"
        if output_log.exists():
            artifact.add_file(str(output_log), name="output.log")
        
        # Add requirements.txt if it exists
        requirements = files_path / "requirements.txt"
        if requirements.exists():
            artifact.add_file(str(requirements), name="requirements.txt")
        
        # Log the artifact
        run.log_artifact(artifact)

        # --------------------------------------------------
        # Reconstruct training history from output.log
        # --------------------------------------------------
        if output_log.exists():
            import re
            print("Replaying training history from output.log – this may take a minute…")
            history_pattern = re.compile(
                r"Epoch\s+(?P<epoch>\d+)\s+\[[^\]]+\]\s+\|\s+Step\s+(?P<step>\d+)\s+\|\s+Loss:\s+(?P<loss>[0-9.]+)\s+\|\s+Avg Loss:\s+(?P<avg_loss>[0-9.]+)\s+\|\s+LR:\s+(?P<lr>[0-9.eE+-]+)")
            val_pattern = re.compile(r"Validation loss:\s+(?P<val_loss>[0-9.]+)")
            step_counter = 0
            with open(output_log, "r") as f_log:
                for line in f_log:
                    m = history_pattern.search(line)
                    if m:
                        step = int(m.group("step"))
                        metrics = {
                            "train_loss": float(m.group("loss")),
                            "avg_train_loss": float(m.group("avg_loss")),
                            "learning_rate": float(m.group("lr")),
                            "epoch": int(m.group("epoch"))
                        }
                        run.log(metrics, step=step, commit=True)
                        step_counter = max(step_counter, step)
                    else:
                        mv = val_pattern.search(line)
                        if mv:
                            # Use next step for validation metric
                            step_counter += 1
                            run.log({"val_loss": float(mv.group("val_loss"))}, step=step_counter, commit=True)
            print("✔️  Finished replaying history")
        else:
            print("No output.log found – skipping history replay")
        
        print(f"✅ Successfully restored run! New run ID: {run.id}")
        print(f"🔗 View at: {run.url}")
        
    finally:
        run.finish()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python restore_wandb_run.py <path_to_cached_run_directory>")
        sys.exit(1)
    
    run_dir = sys.argv[1]
    if not os.path.exists(run_dir):
        print(f"Error: Directory {run_dir} does not exist")
        sys.exit(1)
    
    restore_run(run_dir)