
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable


KNOWN_PIPELINE_STATUSES = {
    "ok",
    "partial_analysis",
    "invalid_input",
    "video_decode_error",
    "no_person_detected",
    "insufficient_pose_quality",
    "unknown_orientation",
    "wrong_exercise",
    "no_reps_detected",
    "invalid_segmentation",
    "invalid_anchors",
    "failed",
}


def run_pipeline_via_subprocess(
    source_path: str,
    exercise: str,
    source_type: str = "video",
    on_progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    pipeline_python = os.getenv("PIPELINE_PYTHON", ".venv/bin/python")
    project_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)
    env["PYTHONUNBUFFERED"] = "1"

    if source_type == "npz":
        command_base = [
            pipeline_python,
            "-u",
            "-m",
            "src.scripts.run_pipeline_json",
            "--npz",
            str(source_path),
            "--exercise",
            str(exercise),
        ]
    else:
        command_base = [
            pipeline_python,
            "-u",
            "-m",
            "src.scripts.run_pipeline_json",
            "--video",
            str(source_path),
            "--exercise",
            str(exercise),
        ]

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        output_json = Path(tmp.name)

    cmd = command_base + ["--output", str(output_json)]

    def _emit(message: str) -> None:
        if on_progress is not None:
            on_progress(message)

    _emit(f"[ui] Entrada: {source_type} | {Path(source_path).name}")
    _emit("[ui] Arrancando proceso Python del pipeline...")
    _emit(
        "[ui] Nota: la primera linea puede tardar (carga de torch/mmpose en el .venv del pipeline)."
    )
    try:
        process = subprocess.Popen(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(project_root),
            env=env,
            bufsize=1,
        )
        output_lines: list[str] = []
        read_done = threading.Event()
        start = time.monotonic()
        last_activity = start

        def _reader() -> None:
            nonlocal last_activity
            if process.stdout is None:
                read_done.set()
                return
            for line in process.stdout:
                clean = line.rstrip("\n\r")
                if not clean:
                    continue
                output_lines.append(clean)
                last_activity = time.monotonic()
                _emit(clean)
            read_done.set()

        reader = threading.Thread(target=_reader, daemon=True)
        reader.start()

        heartbeat = 0
        while True:
            if read_done.is_set() and process.poll() is not None:
                break
            elapsed = int(time.monotonic() - start)
            since_line = int(time.monotonic() - last_activity)
            if since_line >= 3:
                heartbeat += 1
                _emit(
                    f"[ui] Sigue en ejecucion ({elapsed}s). "
                    f"Sin nueva salida desde hace {since_line}s (paso {heartbeat})..."
                )
                last_activity = time.monotonic()
            time.sleep(1.0)

        reader.join(timeout=2.0)
        returncode = process.wait()

        _emit(f"[ui] Proceso terminado (codigo {returncode}). Leyendo resultado...")
        payload = _load_result_json(output_json)
        normalized = _normalize_payload(payload)
        normalized["subprocess_output"] = output_lines
        _emit(f"[ui] Estado final: {normalized.get('status', 'unknown')}")

        if returncode != 0 and normalized.get("status") not in KNOWN_PIPELINE_STATUSES:
            return {
                "status": "failed",
                "ok": False,
                "user_message": "Ha ocurrido un error interno al ejecutar el analisis.",
                "quality": {},
                "repetitions": [],
                "feedback": {},
                "technical_warnings": [],
                "subprocess_output": output_lines,
                "errors": [
                    {
                        "stage": "subprocess",
                        "returncode": returncode,
                        "stdout": "\n".join(output_lines),
                        "stderr": "",
                    }
                ],
                "artifacts": {},
            }
        return normalized
    except Exception as exc:
        _emit(f"[ui] Error en adaptador: {type(exc).__name__}: {exc}")
        return {
            "status": "failed",
            "ok": False,
            "user_message": "No se pudo lanzar el pipeline desde la interfaz.",
            "quality": {},
            "repetitions": [],
            "feedback": {},
            "technical_warnings": [],
            "subprocess_output": [],
            "errors": [{"stage": "adapter", "type": type(exc).__name__, "message": str(exc)}],
            "artifacts": {},
        }
    finally:
        if output_json.exists():
            output_json.unlink(missing_ok=True)


def _load_result_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "failed", "ok": False, "user_message": "No se genero salida JSON del pipeline."}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"status": "failed", "ok": False, "user_message": "JSON de salida invalido."}
    return raw if isinstance(raw, dict) else {"status": "failed", "ok": False, "user_message": "Salida no valida."}


def _normalize_payload(raw: dict[str, Any]) -> dict[str, Any]:
    status = str(raw.get("status", "failed"))
    if status not in KNOWN_PIPELINE_STATUSES:
        status = "failed"
    return {
        "status": status,
        "ok": bool(raw.get("ok", status == "ok")),
        "user_message": str(raw.get("user_message", "")),
        "quality": raw.get("quality", {}) if isinstance(raw.get("quality"), dict) else {},
        "repetitions": raw.get("repetitions", []) if isinstance(raw.get("repetitions"), list) else [],
        "feedback": raw.get("feedback", {}) if isinstance(raw.get("feedback"), dict) else {},
        "technical_warnings": raw.get("technical_warnings", [])
        if isinstance(raw.get("technical_warnings"), list)
        else [],
        "errors": raw.get("errors", []) if isinstance(raw.get("errors"), list) else [],
        "artifacts": raw.get("artifacts", {}) if isinstance(raw.get("artifacts"), dict) else {},
    }
