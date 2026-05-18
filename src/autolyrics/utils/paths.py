"""Project root resolution."""
from pathlib import Path


def get_project_root() -> Path:
    """Return the project root directory (contains pyproject.toml)."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback: walk up until we find a known marker
    return current.parents[4]


PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "data"
CONFIGS_DIR = PROJECT_ROOT / "configs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
