import argparse
import os
from pathlib import Path
from typing import Any, Dict

import yaml

from engine.trainer import Trainer, ExperimentConfig, ModelSpec, OptimizerSpec, DatasetSpec


def _dict_to_obj(cls, d: Dict[str, Any]):
    # recursively construct dataclasses from dicts
    if cls is DatasetSpec:
        return DatasetSpec(**d)
    if cls is ModelSpec:
        return ModelSpec(**d)
    if cls is OptimizerSpec:
        return OptimizerSpec(**d)
    if cls is ExperimentConfig:
        # nested types
        d = dict(d)
        d["model"] = _dict_to_obj(ModelSpec, d["model"]) if isinstance(d.get("model"), dict) else d.get("model")
        d["optimizer"] = _dict_to_obj(OptimizerSpec, d["optimizer"]) if isinstance(d.get("optimizer"), dict) else d.get("optimizer")
        if isinstance(d.get("datasets"), list):
            d["datasets"] = [
                _dict_to_obj(DatasetSpec, x) if isinstance(x, dict) else x for x in d["datasets"]
            ]
        return ExperimentConfig(**d)
    return cls(**d)


def main():
    p = argparse.ArgumentParser(description="Generic YAML-driven training runner")
    p.add_argument("--config", type=str, required=True, help="Path to config.yml")
    args = p.parse_args()

    with open(args.config, "r") as f:
        raw_cfg = yaml.safe_load(f)

    cfg: ExperimentConfig = _dict_to_obj(ExperimentConfig, raw_cfg)

    # Ensure output directory exists
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
