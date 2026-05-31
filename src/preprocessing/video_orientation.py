
from __future__ import annotations

from pathlib import Path

import cv2

from src.utils.paths import OUTPUTS


def flip_video(input_path: Path, output_path: Path) -> Path:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el vídeo: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps_out = fps if fps and fps > 0 else 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Dimensiones inválidas detectadas en: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_out,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"No se pudo crear el vídeo de salida: {output_path}")

    processed = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            flipped = cv2.flip(frame, 1)
            writer.write(flipped)
            processed += 1
    finally:
        cap.release()
        writer.release()

    if processed == 0:
        raise RuntimeError("No se procesó ningún frame; abortando.")
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("El vídeo de salida no se generó correctamente.")
    return output_path


def ensure_video_facing(
    video_path: Path,
    *,
    detected_facing: str,
    target_facing: str = "right",
) -> tuple[Path, bool]:
    if target_facing != "right":
        raise ValueError("Solo se soporta target_facing='right'.")

    if detected_facing == "right":
        return video_path, False
    if detected_facing == "unknown":
        return video_path, False
    if detected_facing == "left":
        flipped_path = OUTPUTS / "tmp" / f"{video_path.stem}_flipped.mp4"
        return flip_video(video_path, flipped_path), True

    raise ValueError(f"detected_facing no soportado: {detected_facing}")
