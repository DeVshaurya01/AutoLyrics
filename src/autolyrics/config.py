"""OmegaConf config loader with project-path resolution."""
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from autolyrics.utils.paths import PROJECT_ROOT, CONFIGS_DIR


def load_config(overrides: list[str] | None = None) -> DictConfig:
    """Load and merge Hydra-style YAML configs, resolve ${project_root} vars."""
    base = OmegaConf.load(CONFIGS_DIR / "base.yaml")

    # Merge sub-configs referenced in the defaults list
    defaults = base.pop("defaults", [])
    merged = OmegaConf.create({})
    for entry in defaults:
        for group, name in entry.items():
            cfg_path = CONFIGS_DIR / group / f"{name}.yaml"
            if cfg_path.exists():
                sub = OmegaConf.load(cfg_path)
                merged = OmegaConf.merge(merged, sub)

    cfg = OmegaConf.merge(base, merged)

    # Resolve ${project_root} placeholder
    OmegaConf.register_new_resolver(
        "project_root", lambda: str(PROJECT_ROOT), replace=True
    )

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    return cfg
