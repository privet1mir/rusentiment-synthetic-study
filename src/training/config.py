from pathlib import Path
from typing import Optional, Union

import yaml
from omegaconf import OmegaConf
from pydantic import BaseModel

from const import PROJECT_ROOT


class DataConfig(BaseModel):
    dataset_path: Path = PROJECT_ROOT / "dataset"

    train_file: Optional[Path] = None
    val_file: Optional[Path] = None
    test_file: Optional[Path] = None

    max_length: int = 128
    num_classes: int = 3


class ModuleConfig(BaseModel):
    model_name: str = "DeepPavlov/rubert-base-cased"


class TrainerConfig(BaseModel):

    output_dir: Path = PROJECT_ROOT / "trained_models"

    learning_rate: float = 2e-5
    num_train_epochs: int = 20

    per_device_train_batch_size: int = 128
    per_device_eval_batch_size: int = 128

    weight_decay: float = 0.01

    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"

    logging_dir: Path = PROJECT_ROOT / "logs"
    logging_steps: int = 50

    load_best_model_at_end: bool = True
    metric_for_best_model: str = "macro_f1"

    seed: int = 42


class ExperimentConfig(BaseModel):

    project_name: str = "rusentiment_synthetic_study"
    experiment_name: str = "RuBERT Baseline Natural Distribution 1k"

    trainer: TrainerConfig = TrainerConfig()
    data: DataConfig = DataConfig()
    module: ModuleConfig = ModuleConfig()

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ExperimentConfig":

        cfg = OmegaConf.to_container(
            OmegaConf.load(path),
            resolve=True
        )

        return cls(**cfg)

    def to_yaml(self, path: Union[str, Path]) -> None:

        with open(path, "w") as f:
            yaml.safe_dump(
                self.model_dump(),
                f,
                default_flow_style=False,
                sort_keys=False
            )