
from __future__ import annotations

import argparse
from pathlib import Path

from src.visualization import render_pose_frame_range_overlay

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def _ask_bool(prompt: str, default: bool) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    raw = input(f"{prompt} {suffix}: ").strip().lower()
    if raw == "":
        return default
    return raw in {"y", "yes", "s", "si"}


def _list_available_videos() -> list[Path]:
    data_root = Path("data")
    if not data_root.is_dir():
        return []
    videos = [p for p in data_root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    return sorted(videos)


def _select_video_interactively() -> Path:
    videos = _list_available_videos()
    if not videos:
        raw = input("No se encontraron videos en data/. Introduce ruta manual: ").strip()
        path = Path(raw).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Video inexistente: {path}")
        return path
    print("\nVideos disponibles (data/):")
    for idx, path in enumerate(videos, start=1):
        print(f"{idx}. {path.as_posix()}")
    while True:
        raw = input("\nSelecciona numero de video: ").strip()
        try:
            choice = int(raw)
        except ValueError:
            print("Entrada invalida. Introduce un numero.")
            continue
        if 1 <= choice <= len(videos):
            return videos[choice - 1]
        print(f"Numero fuera de rango (1..{len(videos)}).")


def _list_available_npz() -> list[Path]:
    outputs_root = Path("outputs")
    if not outputs_root.is_dir():
        return []
    npzs = [p for p in outputs_root.rglob("*.npz") if p.is_file()]
    return sorted(npzs)


def _select_npz_interactively() -> Path:
    npzs = _list_available_npz()
    if not npzs:
        raw = input("No se encontraron NPZ en outputs/. Introduce ruta manual: ").strip()
        path = Path(raw).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"NPZ inexistente: {path}")
        return path
    print("\nNPZ disponibles (outputs/):")
    for idx, path in enumerate(npzs, start=1):
        print(f"{idx}. {path.as_posix()}")
    while True:
        raw = input("\nSelecciona numero de NPZ: ").strip()
        try:
            choice = int(raw)
        except ValueError:
            print("Entrada invalida. Introduce un numero.")
            continue
        if 1 <= choice <= len(npzs):
            return npzs[choice - 1]
        print(f"Numero fuera de rango (1..{len(npzs)}).")


def _video_stem_to_npz_candidates(video_stem: str) -> list[Path]:
    outputs_root = Path("outputs")
    if not outputs_root.is_dir():
        return []
    ranked_names = [
        f"{video_stem}_rtmpose_clean.npz",
        f"{video_stem}_rtmpose.npz",
        f"{video_stem}_rtmpose_probe.npz",
    ]
    all_npzs = [p for p in outputs_root.rglob("*.npz") if p.is_file()]
    ranked: list[Path] = []
    for name in ranked_names:
        ranked.extend(sorted([p for p in all_npzs if p.name == name]))
    return ranked


def _npz_to_video_stem(npz_path: Path) -> str:
    stem = npz_path.stem
    for suffix in ("_rtmpose_clean", "_rtmpose_probe", "_rtmpose"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _video_candidates_from_stem(stem: str) -> list[Path]:
    return sorted([p for p in _list_available_videos() if p.stem == stem])


def _ask_tramos() -> list[tuple[int, int]]:
    num = int(input("Cuantos tramos quieres generar: ").strip())
    if num < 1:
        raise ValueError("El numero de tramos debe ser >= 1.")
    tramos: list[tuple[int, int]] = []
    for i in range(num):
        start = int(input(f"Tramo {i+1} - Frame inicial: ").strip())
        end = int(input(f"Tramo {i+1} - Frame final: ").strip())
        if start > end:
            raise ValueError(f"Tramo {i+1}: inicio > final ({start} > {end}).")
        tramos.append((start, end))
    return tramos


def _resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path, list[tuple[int, int]], int, str]:
    if args.video is not None and args.npz is not None and args.start is not None and args.end is not None:
        return (
            Path(args.video).expanduser(),
            Path(args.npz).expanduser(),
            [(int(args.start), int(args.end))],
            int(args.step),
            str(args.pose_source),
        )

    print("\nSelecciona modo de entrada:")
    print("1. Elegir video y resolver NPZ por nombre")
    print("2. Elegir NPZ y resolver video por nombre")
    mode = input("Opcion [1/2]: ").strip()
    if mode not in {"1", "2"}:
        raise ValueError("Opcion invalida. Debe ser 1 o 2.")

    if mode == "1":
        video = _select_video_interactively()
        npz_candidates = _video_stem_to_npz_candidates(video.stem)
        if not npz_candidates:
            raise FileNotFoundError(f"No se encontro NPZ asociado para video '{video.stem}' en outputs/.")
        npz = npz_candidates[0]
    else:
        npz = _select_npz_interactively()
        stem = _npz_to_video_stem(npz)
        videos = _video_candidates_from_stem(stem)
        if not videos:
            raise FileNotFoundError(f"No se encontro video en data/ para NPZ '{npz.name}' (stem='{stem}').")
        video = videos[0]

    tramos = _ask_tramos()
    step = int(input("Step [1]: ").strip() or "1")
    pose_source = (input("Pose source [clean/raw/auto] (default auto): ").strip() or "auto").lower()
    return video, npz, tramos, step, pose_source


def main() -> None:
    parser = argparse.ArgumentParser(description="Render pose overlays over real video frames.")
    parser.add_argument("--video", type=str, default=None, help="Video path.")
    parser.add_argument("--npz", type=str, default=None, help="Associated NPZ path.")
    parser.add_argument("--start", type=int, default=None, help="Start frame.")
    parser.add_argument("--end", type=int, default=None, help="End frame.")
    parser.add_argument("--step", type=int, default=1, help="Frame step.")
    parser.add_argument(
        "--pose-source",
        choices=["clean", "raw", "auto"],
        default="auto",
        help="Pose source inside NPZ.",
    )
    parser.add_argument("--score-threshold", type=float, default=0.30, help="Visibility score threshold.")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory.")
    args = parser.parse_args()

    video, npz, tramos, step, pose_source = _resolve_inputs(args)
    if args.out_dir is None:
        out_dir = Path("outputs/visualization/debug/pose_frames") / video.stem / pose_source
    else:
        out_dir = Path(args.out_dir).expanduser()

    results = []
    for start, end in tramos:
        tramo_results = render_pose_frame_range_overlay(
            video_path=video,
            npz_path=npz,
            start_frame=start,
            end_frame=end,
            step=step,
            output_dir=out_dir,
            pose_source=pose_source,
            score_threshold=float(args.score_threshold),
            draw_labels=True,
            draw_scores=False,
            highlight_right_chain=True,
        )
        results.extend(tramo_results)

    actual_source = results[0]["pose_source"] if results else pose_source
    print(f"video={video.as_posix()}")
    print(f"npz={npz.as_posix()}")
    print(f"tramos={tramos} step={step}")
    print(f"pose_source={actual_source}")
    print(f"output_dir={out_dir.as_posix()}")
    print(f"num_images={len(results)}")


if __name__ == "__main__":
    main()

