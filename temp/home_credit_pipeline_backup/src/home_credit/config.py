from dataclasses import dataclass
from typing import Any, Dict
import yaml


@dataclass
class Config:
    """Configuration object loaded from YAML."""
    data: Dict[str, Any]
    preprocessing: Dict[str, Any]
    model: Dict[str, Any]
    validation: Dict[str, Any]
    output: Dict[str, Any]

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        return cls(**cfg)
