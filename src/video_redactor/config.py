from dataclasses import dataclass, field
from typing import List

@dataclass
class MatchConfig:
    mode: str = "patterns"
    redact_types: List[str] = field(default_factory=lambda: ["email", "phone", "keywords"])
    custom_keywords: List[str] = field(default_factory=list)

@dataclass
class ProcessingConfig:
    input_path: str
    match_config: MatchConfig
    output_path: str = ""
    frame_skip: int = 1
    padding: int = 25
    blur_strength: int = 35
    max_ocr_dim: int = 1080
    static_thresh: float = 0.8
    change_thresh: float = 1.0
    mag_ratio: float = 1.0
    min_size: int = 10
    threads: int = 4
    no_contrast: bool = False
    no_audio: bool = False
    workers: str = "1"
    cpu: bool = False
    debug: bool = False
