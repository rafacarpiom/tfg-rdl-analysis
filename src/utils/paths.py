
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
CONFIGS = ROOT / "configs"
WEIGHTS = ROOT / "weights"
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
REFERENCES = ROOT / "references"

# Nota: en este proyecto los vídeos se guardan directamente en data/.
# Mantener `VIDEOS` como alias permite que los CLIs sigan usando la misma
# interfaz sin cambiar su lógica interna.
VIDEOS = DATA
NPZ_OUTPUTS = OUTPUTS / "npz"

# Referencias: activos estables (no se guardan bajo outputs/).
RDL_REFERENCES = REFERENCES / "rdl"


def rdl_reference_dir(reference_name: str = "PM-Ideal") -> Path:
    preferred = (RDL_REFERENCES / reference_name).resolve()
    if preferred.is_dir():
        return preferred
    legacy = (OUTPUTS / "references" / "rdl" / reference_name).resolve()
    if legacy.is_dir():
        return legacy
    return preferred

RTMPOSE_CONFIG = CONFIGS / "rtmpose-l_8xb256-420e_coco-256x192.py"
RTMPOSE_CHECKPOINT = WEIGHTS / "rtmpose-l_simcc-coco_pt-aic-coco_420e-256x192-1352a4d2_20230127.pth"
YOLO_WEIGHTS = WEIGHTS / "yolov8s.pt"


def ensure_runtime_dirs() -> None:
    NPZ_OUTPUTS.mkdir(parents=True, exist_ok=True)
    RDL_REFERENCES.mkdir(parents=True, exist_ok=True)

