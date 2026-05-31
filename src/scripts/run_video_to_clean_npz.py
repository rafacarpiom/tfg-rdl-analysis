
from __future__ import annotations

import argparse
from pathlib import Path

from src.pose.extraction import extract_video_to_npz
from src.pose_cleaning import clean_pose_npz

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def _list_available_videos() -> list[Path]:
    data_root = Path("data")
    if not data_root.is_dir():
        return []
    videos = [p for p in data_root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    return sorted(videos)


def _ask_video_path() -> Path:
    videos = _list_available_videos()
    if not videos:
        raw = input("No se encontraron videos en data/. Introduce ruta manual: ").strip()
        path = Path(raw).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Ruta de video inexistente: {path}")
        return path

    print("\nVideos disponibles (data/):")
    for i, p in enumerate(videos, start=1):
        print(f"{i}. {p.as_posix()}")

    while True:
        raw = input("Selecciona numero de video: ").strip()
        try:
            idx = int(raw)
        except ValueError:
            print("Entrada invalida. Introduce un numero.")
            continue
        if 1 <= idx <= len(videos):
            return videos[idx - 1]
        print(f"Numero fuera de rango (1..{len(videos)}).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pose extraction and cleaning to NPZ.")
    parser.add_argument("--video", type=str, default=None, help="Path to input video. If omitted, interactive selection.")
    parser.add_argument("--out-dir", type=str, default="outputs/npz", help="Directory to save raw and clean NPZ.")
    parser.add_argument("--keep-raw", action="store_true", help="Keep intermediate raw NPZ file.")
    parser.add_argument("--quiet", action="store_true", help="Reduce extraction verbosity.")
    args = parser.parse_args()

    video_path = Path(args.video).expanduser() if args.video else _ask_video_path()
    if not video_path.is_file():
        raise FileNotFoundError(f"Ruta de video inexistente: {video_path}")

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_npz = out_dir / f"{video_path.stem}_rtmpose.npz"
    clean_npz = out_dir / f"{video_path.stem}_rtmpose_clean.npz"

    extract_video_to_npz(
        video_path=str(video_path),
        output_path=str(raw_npz),
        verbose=not args.quiet,
    )
    clean_pose_npz(str(raw_npz), str(clean_npz))

    if not args.keep_raw and raw_npz.is_file():
        raw_npz.unlink()
    elif raw_npz.is_file():
        print(f"raw_npz: {raw_npz.as_posix()}")
    print(f"clean_npz: {clean_npz.as_posix()}")


if __name__ == "__main__":
    main()

