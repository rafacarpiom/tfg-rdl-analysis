# `pipeline`

Orquestación del **análisis completo en memoria** desde vídeo (o continuación desde pose ya calculada vía scripts).

## Módulos principales

| Fichero            | Rol |
|--------------------|-----|
| `full_analysis.py` | Flujo end-to-end RDL |
| `config.py`        | `FullAnalysisConfig`, pose, orientación, referencia |
| `results.py`       | `PipelineResult`, serialización para CLI/UI |
| `validation.py`    | Validación de vídeo y contratos de datos |
| `status.py`        | Constantes de estado (`ok`, `no_reps_detected`, …) |
| `contracts.py`     | Comprobaciones de forma de pose/segmentación/anclas |

API pública: `from src.pipeline import run_full_analysis, FullAnalysisConfig, …`

## Flujo (`run_full_analysis`)

```text
vídeo original
  → validación de entrada
  → pose raw (YOLO + RTMPose, probe + pasada final)
  → orientación lateral y flip si el sujeto mira a la izquierda
  → pose clean (pose_cleaning)
  → segmentación RDL
  → normalización de secuencia (usuario)
  → analysis_context (usuario + referencia PM-Ideal)
  → detectores RDL (bar_far, arms, asymmetry, hip_hinge, …)
  → normalización de evidencias → agregación → informe de feedback
  → PipelineResult / dict para CLI, tests o UI
```

Todo ocurre **en memoria**: `full_analysis` no escribe NPZ/JSON/PNG por defecto.

## Paquetes colaboradores

- **`src/pose`**: detección, selección de persona, RTMPose, orientación.
- **`src/pose_cleaning`**: filtrado, interpolación, suavizado (raw → clean).
- **`src/biomechanics/rdl/segmentation`**: repeticiones y anclas (core sin plots).
- **`src/biomechanics/rdl/analysis_context`**: emparejado usuario–ideal y caché de anclas.
- **`src/biomechanics/rdl/detectors`**: métricas y reglas por error técnico.
- **`src/biomechanics/rdl/feedback`**: evidencia, agregación y texto de feedback.
- **`src/visualization/rdl`**: exports debug (PNG/JSON); opcional, desde scripts.

## Puntos de entrada habituales

| Script | Uso |
|--------|-----|
| `src.scripts.run_pipeline_json` | Subproceso de la UI Flet; escribe JSON |
| `src.scripts.run_pipeline` | CLI interactivo; puede guardar segmentación y bundles debug |
| `run_full_analysis` directo | Tests e integraciones Python |

## Referencia ideal

- Directorio por defecto: `references/rdl/PM-Ideal/` (configurable en `FullAnalysisConfig.reference_dir`).
- Carga en `src/biomechanics/rdl/analysis_context/reference_loader.py`.
- La comparación cuantitativa usuario vs ideal está integrada en detectores y contexto; no es un módulo aparte.

## Aclaraciones

- Los estados `invalid_input`, `no_person_detected`, `no_reps_detected`, etc. se devuelven como resultado estructurado; no dependen de excepciones no controladas.
- `--exercise` en `run_pipeline_json` está reservado; el backend actual es **solo RDL**.
- Persistencia en disco (NPZ, plots, `debug_runs/`) la deciden los scripts, no `full_analysis`.
