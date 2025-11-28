from __future__ import annotations
from dataclasses import dataclass


@dataclass
class AdapterConfig:
    WIDTH: int                # CLIP hidden dim (예: 1024)
    ADAPTER_SCALE: float = 0.25
    HEADS: int = 4
    GROUPS: int = 4


@dataclass
class DataConfig:
    NUM_INPUT_FRAMES: int = 12   # 비디오 T (stage-2 기준)


@dataclass
class D2STConfig:
    ADAPTER: AdapterConfig
    DATA: DataConfig
