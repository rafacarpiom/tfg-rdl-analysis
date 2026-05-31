
from __future__ import annotations

import argparse
from pathlib import Path

from src.biomechanics.rdl import run_rdl_segmentation
from src.visualization.rdl import plot_rdl_segmentation_debug


def _list_clean_npz_candidates() -> list[Path]:
    base = Path("outputs/npz")
    if not base.is_dir():
        return []
    candidates = [p for p in base.rglob("*.npz") if p.is_file() and p.stem.endswith("_clean")]
    return sorted(candidates)


def _select_npz_interactively() -> Path:
    candidates = _list_clean_npz_candidates()
    if not candidates:
        raw = input("No se encontraron *_clean.npz en outputs/npz. Introduce ruta manual: ").strip()
        path = Path(raw).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"NPZ inexistente: {path}")
        return path

    print("\nNPZ clean disponibles (outputs/npz):")
    for idx, path in enumerate(candidates, start=1):
        print(f"{idx}. {path.as_posix()}")

    while True:
        raw = input("\nSelecciona numero de NPZ: ").strip()
        try:
            choice = int(raw)
        except ValueError:
            print("Entrada invalida. Introduce un numero.")
            continue
        if 1 <= choice <= len(candidates):
            return candidates[choice - 1]
        print(f"Numero fuera de rango (1..{len(candidates)}).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RDL segmentation and generate only debug PNG.",
    )
    parser.add_argument("--npz", type=str, default=None, help="Path to pose NPZ (raw or clean).")
    parser.add_argument("--out", type=str, default=None, help="Output directory.")
    args = parser.parse_args()

    npz_path = Path(args.npz).expanduser() if args.npz else _select_npz_interactively()
    result = run_rdl_segmentation(npz_path=npz_path)
    out_dir = (
        Path(args.out)
        if args.out is not None
        else Path("outputs/segmentation/rdl") / str(result["video_id"])
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = str(result["video_id"]) + "_rdl_segmentation"
    png_path = plot_rdl_segmentation_debug(result, out_dir / f"{stem}.png")

    print(f"video_id={result['video_id']}")
    print(f"pose_source={result['pose_source']}")
    print(f"has_clean_pose={result['has_clean_pose']}")
    print(f"segmentation_status={result['segmentation_status']}")
    print(f"num_reps={result['summary']['num_reps']}")
    print(f"num_reps_with_valid_anchors={result['summary'].get('num_reps_with_valid_anchors', 0)}")
    print(f"num_reps_with_invalid_anchors={result['summary'].get('num_reps_with_invalid_anchors', 0)}")
    print(f"anchor_method={result['summary'].get('anchor_method', 'signal_progress')}")
    print(f"plot_path={png_path.as_posix()}")
    print(f"output_dir={out_dir.as_posix()}")


if __name__ == "__main__":
    main()

