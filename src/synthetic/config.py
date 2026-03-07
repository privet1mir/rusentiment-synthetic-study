from pathlib import Path
from typing import Optional, Union

import yaml
from omegaconf import OmegaConf
from pydantic import BaseModel

from const import PROJECT_ROOT, LABELS, LABELS_DISTRIBUTION

class GenerateConfig(BaseModel):
    dataset_path: Path = PROJECT_ROOT / "dataset"

    train_file: Optional[Path] = None
    val_file: Optional[Path] = None
    test_file: Optional[Path] = None

    num_samples: int = 1000
    labels: list = LABELS
    labels_distribution: dict = LABELS_DISTRIBUTION

    deduplication: bool = False

    provider: str = "OpenAI" 
    model: str = "gpt-5-mini-2025-08-07"



class ExperimentConfig(BaseModel):

    project_name: str = "rusentiment_synthetic_study"
    experiment_name: str = "Generate Synthetic Raw 1k"
    prompt_type: str = "base"

    generator: GenerateConfig = GenerateConfig()

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