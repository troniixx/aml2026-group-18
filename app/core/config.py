"""
core/config.py
Loads config.yaml and exposes a single typed Config object.
All other modules import from here — never read the YAML directly.
"""
from __future__ import annotations
 
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
 
import yaml
 
# Root of the project (naruto_app/)
APP_ROOT = Path(__file__).parent.parent
 
 
# ── YAML constructor: !path ───────────────────────────────────────────────────
# Allows writing cross-platform paths in config.yaml using forward slashes:
#
#   assets_dir: !path C:/Users/babus/projects/assets       # Windows & Mac
#   assets_dir: !path /home/babus/projects/assets          # Mac/Linux
#
# Python's Path() normalises the separator for the current OS automatically,
# so the same config.yaml works on both Windows and Mac without any changes.
def _path_constructor(loader, node) -> Path:
    raw = loader.construct_scalar(node)
    return Path(raw)          # Path handles / and \ on all platforms
 
yaml.add_constructor("!path", _path_constructor, Loader=yaml.SafeLoader)
 
 
@dataclass
class ModelConfig:
    name:                 str
    script_path:          Path    # {model}_live.py  — defines build_model()
    weights_path:         Path    # {model}_best.pth — trained weights
    confidence_threshold: float   # minimum confidence to accept a prediction
 
 
@dataclass
class UIConfig:
    panel_width:     int
    seal_thumb_size: int
    camera_index:    int
    ghost_opacity:   float
    primary_color:   tuple[int, int, int]
    secondary_color: tuple[int, int, int]
 
 
@dataclass
class Config:
    models:          dict[str, ModelConfig]
    active_model:    str
    ui:              UIConfig
    seal_images:     dict[str, Optional[Path]]
    ghost_images:    dict[str, Optional[Path]]
    jutsu_combos:    dict[str, list[str]]
    all_seals:       list[str]              = field(default_factory=list)
    mediapipe_model: Optional[Path]         = None
 
    @property
    def active_model_config(self) -> ModelConfig:
        return self.models[self.active_model]
 
 
def _resolve(path_str: Optional[str | Path], base: Optional[Path] = None) -> Optional[Path]:
    """
    Resolve a path to an absolute Path, returning None if input is None.
 
    Accepts either a plain string or a Path object (returned by the !path tag).
    Resolution order:
      1. None              → None
      2. Absolute path     → used as-is
      3. Relative path     → resolved against *base* (if given), else APP_ROOT
    """
    if path_str is None:
        return None
    p = Path(path_str)          # no-op if already a Path
    if not p.is_absolute():
        p = (base or APP_ROOT) / p
    if not p.exists():
        print(f"[WARN] Config path does not exist: {p}")
    return p
 
 
def load(config_path: Path | str | None = None) -> Config:
    """
    Load and validate config.yaml.
 
    Parameters
    ----------
    config_path : path to the YAML file.
                  Defaults to naruto_app/config.yaml.
    """
    if config_path is None:
        config_path = APP_ROOT / "config.yaml"
 
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
 
    # ── MediaPipe hand landmarker model ──────────────────────────────────────
    mediapipe_model: Optional[Path] = _resolve(raw.get("mediapipe_model"))
 
    # ── Models ────────────────────────────────────────────────────────────────
    models: dict[str, ModelConfig] = {}
    for key, m in raw["models"].items():
        models[key] = ModelConfig(
            name=m["name"],
            script_path=_resolve(m["script"]),
            weights_path=_resolve(m["weights"]),
            confidence_threshold=float(m.get("confidence_threshold", 0.5)),
        )
 
    active_model = raw["active_model"]
    if active_model not in models:
        raise ValueError(
            f"active_model '{active_model}' not found in models. "
            f"Available: {list(models)}"
        )
 
    # ── UI ────────────────────────────────────────────────────────────────────
    u = raw["ui"]
    ui = UIConfig(
        panel_width=int(u["panel_width"]),
        seal_thumb_size=int(u["seal_thumb_size"]),
        camera_index=int(u["camera_index"]),
        ghost_opacity=float(u["ghost_opacity"]),
        primary_color=tuple(u["primary_color"]),
        secondary_color=tuple(u["secondary_color"]),
    )
 
    # ── Assets directory (optional base for seal/ghost image paths) ─────────────
    assets_dir: Optional[Path] = _resolve(raw.get("assets_dir"))
 
    # ── Seal & ghost image paths ──────────────────────────────────────────────
    # Relative paths are resolved against assets_dir first, then APP_ROOT.
    active_seal_key = raw.get("active_seal_images", "seal_images")
    seal_images: dict[str, Optional[Path]] = {
        seal: _resolve(path, base=assets_dir)
        for seal, path in raw[active_seal_key].items()
    }
    
    ghost_images: dict[str, Optional[Path]] = {
        seal: _resolve(path, base=assets_dir)
        for seal, path in raw["ghost_images"].items()
    }
 
    all_seals = list(seal_images.keys())
 
    # ── Jutsu combos ──────────────────────────────────────────────────────────
    jutsu_combos: dict[str, list[str]] = raw.get("jutsu_combos", {})
 
    return Config(
        models=models,
        active_model=active_model,
        ui=ui,
        seal_images=seal_images,
        ghost_images=ghost_images,
        jutsu_combos=jutsu_combos,
        all_seals=all_seals,
        mediapipe_model=mediapipe_model,
    )