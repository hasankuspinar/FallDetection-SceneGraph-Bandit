from dataclasses import dataclass
from typing import Any, Dict
import yaml

@dataclass
class Config:
    raw: Dict[str, Any]

def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(raw=raw)

def cfg_get(cfg: Config, *keys, default=None):
    d = cfg.raw
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d
