# TFG — Análisis biomecánico de RDL desde vídeo (RTMPose)

Sistema de estimación de pose 2D (YOLO + RTMPose), limpieza de keypoints, segmentación de repeticiones de **peso muerto rumano (RDL)**, comparación contra una referencia ideal (**PM-Ideal**) y detectores de técnica con **informe de feedback**. Incluye interfaz de escritorio con **Flet**.

## Objetivo

Extraer series temporales de keypoints desde vídeo, analizar la ejecución del RDL frente a un patrón ideal y devolver estados estructurados (pipeline) y mensajes de feedback (UI y CLI).

## Estructura del proyecto

```text
tfg-rdl-entrega/
  src/
    app/                    # UI Flet (Technique Coach)
    pipeline/               # Orquestación: run_full_analysis
    pose/                   # Detección, RTMPose, extracción a NPZ
    pose_cleaning/          # Raw → clean
    preprocessing/          # Orientación / flip de vídeo
    biomechanics/
      normalization/        # Normalización de secuencias
      rdl/                  # Segmentación, contexto, detectores, feedback
    visualization/          # Debug PNG (fuera del core)
    scripts/                # CLIs (pipeline, UI, debug, evaluación)
    utils/
  configs/                  # Config RTMPose (OpenMMLab)
  references/rdl/PM-Ideal/  # Referencia ideal (JSON + NPZ; ver abajo)
  data/                     # Vídeo de ejemplo versionado; otros .mp4 locales ignorados por Git
  weights/                  # Pesos YOLO/RTMPose (no versionados)
  outputs/                  # NPZ, debug, resultados (no versionados)
  tests/
  corpus_config.json        # Casos para run_corpus_evaluation.py
  requirements.txt          # Entorno .venv (pipeline)
  requirements_flet.txt     # Entorno .venv_flet (UI)
```

## Requisitos previos

Antes de empezar, asegúrate de tener:

- **Python 3.10**
- **Git**
- **Linux o WSL2 (recomendado)**: OpenMMLab (`mmcv` / `mmpose`) suele dar menos problemas en este entorno.
- **GPU NVIDIA (opcional)**: acelera bastante la inferencia. Sin GPU el pipeline también funciona, pero más lento.
- **Espacio en disco libre**: ~**6 GB** (entornos virtuales, dependencias y modelos).

### Instalar Python 3.10 y Git (Ubuntu / WSL)

```bash
sudo apt update
sudo apt install -y software-properties-common git
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3.10-dev
```

Comprobación rápida:

```bash
python3.10 --version
git --version
```

## Clonar el repositorio

```bash
git clone https://github.com/rafacarpiom/tfg-rdl-analysis.git
cd tfg-rdl-analysis
```

## Crear los dos entornos virtuales

Este proyecto usa **dos entornos separados**:

| Entorno      | Fichero                 | Uso                               |
|--------------|-------------------------|-----------------------------------|
| `.venv`      | `requirements.txt`      | Pipeline (torch, mmpose, YOLO…)   |
| `.venv_flet` | `requirements_flet.txt` | Interfaz Flet (sin stack pesado)  |

> **No mezcles** paquetes entre ambos entornos.

### 1) Entorno del pipeline (`.venv`)

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Paso 1: instalar PyTorch con soporte CUDA 12.1
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# Paso 2: instalar openmim para gestionar paquetes OpenMMLab
pip install openmim==0.3.9

# Paso 3: instalar componentes OpenMMLab con mim
mim install mmengine==0.10.7
mim install mmcv==2.1.0
mim install mmdet==3.3.0

# Paso 4: instalar mmpose sin chumpy (dependencia abandonada no usada)
pip install mmpose==1.3.2 --no-deps

# Paso 5: instalar el resto de dependencias
pip install -r requirements.txt --no-deps

deactivate
```

> Si no tienes GPU CUDA, sustituye el paso 1 por:
> `pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu`
> El pipeline funcionará más lento pero sin cambios en el código.

### 2) Entorno de la interfaz (`.venv_flet`)

```bash
python3.10 -m venv .venv_flet
source .venv_flet/bin/activate
python -m pip install --upgrade pip
pip install -r requirements_flet.txt
deactivate
```

## Descargar los pesos de los modelos

Los pesos **no se incluyen en Git**. Deben estar en `weights/`. Rutas por defecto (`src/utils/paths.py`):

- `configs/rtmpose-l_8xb256-420e_coco-256x192.py`
- `weights/yolov8s.pt`
- `weights/rtmpose-l_simcc-coco_pt-aic-coco_420e-256x192-1352a4d2_20230127.pth`

### Opción A — Automática (recomendada)

Con `.venv` activado:

```bash
source .venv/bin/activate

mkdir -p weights
# Descarga YOLOv8s (se descarga en el directorio actual)
python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"
# Muévelo a weights/
mv yolov8s.pt weights/

# Descarga RTMPose-L (requiere openmim en el entorno)
mim download mmpose --config rtmpose-l_8xb256-420e_coco-256x192 --dest weights/

deactivate
```

> Si yolov8s.pt no aparece en weights/ tras el comando anterior, descárgalo manualmente desde https://github.com/ultralytics/assets/releases y colócalo en weights/

Si `mim` no está en el PATH, prueba: `python -m mim download mmpose --config rtmpose-l_8xb256-420e_coco-256x192 --dest weights/`

### Opción B — Manual

1. **YOLOv8s**: descarga `yolov8s.pt` desde [releases de Ultralytics](https://github.com/ultralytics/assets/releases) y colócalo en `weights/`.
2. **RTMPose-L**: descarga `rtmpose-l_simcc-coco_pt-aic-coco_420e-256x192-1352a4d2_20230127.pth` desde el model zoo de OpenMMLab y colócalo en `weights/`.

### Verificar pesos

Debes tener exactamente:

- `weights/yolov8s.pt`
- `weights/rtmpose-l_simcc-coco_pt-aic-coco_420e-256x192-1352a4d2_20230127.pth`

```bash
ls -lh weights/yolov8s.pt
ls -lh weights/rtmpose-l_simcc-coco_pt-aic-coco_420e-256x192-1352a4d2_20230127.pth
```

> Nota: `mim download` también descarga un fichero `.py` de configuración en `weights/`. Puedes ignorarlo; el fichero de configuración que usa el pipeline está en `configs/`.

Si el fichero RTMPose tiene otro nombre tras la descarga, renómbralo al nombre anterior.

## Verificar la instalación

### GPU disponible (opcional)

```bash
source .venv/bin/activate
python -c "import torch; print('CUDA disponible:', torch.cuda.is_available())"
deactivate
```

### Import de mmpose

```bash
source .venv/bin/activate
python -c "import mmpose; print('mmpose OK')"
deactivate
```

## Lanzar la aplicación

```bash
source .venv_flet/bin/activate
export PIPELINE_PYTHON="$PWD/.venv/bin/python"
PYTHONPATH=. python -m src.scripts.run_app
```

Abre en el navegador: **http://127.0.0.1:8550**

La UI invoca el pipeline en subproceso (`src.scripts.run_pipeline_json`). La **primera ejecución** puede tardar más porque carga `torch` y `mmpose`.

## Probar con el vídeo de ejemplo

El repositorio incluye `data/ejemplo_rdl.mp4` (~5 MB, grabación lateral de RDL con error de flexión de espalda).

### Desde la interfaz

1. Lanza la app (sección anterior).
2. Selecciona `data/ejemplo_rdl.mp4`.
3. Ejecuta el análisis y espera a que termine.

### Alternativa por CLI

```bash
source .venv/bin/activate
PYTHONPATH=. python -m src.scripts.run_pipeline --video data/ejemplo_rdl.mp4
```

## Referencia PM-Ideal

El análisis completo necesita la carpeta `references/rdl/PM-Ideal/` con al menos:

- `ideal_segmentation_result.json`
- `ideal_pose_sequence_normalized_meta.json`
- `ideal_pose_sequence_normalized.npz` (obligatorio para `load_rdl_reference`)

Si falta el NPZ normalizado, genera la referencia a partir del NPZ clean del ideal:

```bash
source .venv/bin/activate
PYTHONPATH=. python -m src.scripts.run_rdl_ideal_sequence_normalization \
  --clean-npz references/rdl/PM-Ideal/ideal_rtmpose_clean.npz \
  --out-dir references/rdl/PM-Ideal
```

En el repositorio ya vienen versionados los JSON y `ideal_pose_sequence_normalized.npz`. Otros `.npz` grandes pueden estar en `.gitignore`.

## Ejecución

### Análisis completo RDL (vídeo → feedback en memoria)

Desde Python:

```bash
source .venv/bin/activate
PYTHONPATH=. python -c "
from pathlib import Path
from src.pipeline import run_full_analysis, FullAnalysisConfig
r = run_full_analysis(FullAnalysisConfig(video_path=Path('data/tu_video.mp4')))
print(r.get('pipeline_result', {}).get('status'))
"
```

### CLI interactivo (guardado opcional en disco, debug)

```bash
source .venv/bin/activate
PYTHONPATH=. python -m src.scripts.run_pipeline --video data/tu_video.mp4
```

### Salida JSON para la UI o integraciones

```bash
PYTHONPATH=. python -m src.scripts.run_pipeline_json \
  --video data/tu_video.mp4 \
  --output /tmp/resultado.json
```

También admite `--npz` con un NPZ **clean** ya generado.

### Solo pose → NPZ (sin RDL)

```bash
PYTHONPATH=. python -m src.scripts.run_video_to_clean_npz \
  --video data/tu_video.mp4 \
  --output outputs/npz/tu_video_rtmpose_clean.npz
```

### Segmentación RDL desde NPZ clean

```bash
PYTHONPATH=. python -m src.scripts.run_rdl_segmentation \
  --npz outputs/npz/tu_video_rtmpose_clean.npz \
  --out outputs/segmentation/tu_video
```

### Evaluación del corpus de pruebas

```bash
PYTHONPATH=. python run_corpus_evaluation.py --config corpus_config.json
```

Requiere vídeos bajo `data/tests/` (subcarpetas `funcional`, `baseline`, `robustness`) y, opcionalmente, NPZ en caché en `outputs/npz/clear/`.

> Los vídeos del corpus no están incluidos en el repositorio.
> Si tienes los vídeos, colócalos en:
> `data/tests/funcional/`, `data/tests/baseline/`, `data/tests/robustness/`

## Formato NPZ (pose)

Tras extracción / cleaning típico:

- `kps_xy` / `kps_xy_clean`: `(T, 17, 2)` en píxeles
- `kps_score` / scores asociados
- `bbox_xyxy`, `frame_idx`, `fps`, metadatos en `meta` o campos equivalentes

## Tests

```bash
source .venv/bin/activate
PYTHONPATH=. python -m pytest tests/ -q
```

## Herramientas de diagnóstico

`src/scripts/` incluye 12 scripts de diagnóstico interactivo 
(`run_*_debug.py`), uno por cada detector técnico RDL. Cada script 
permite inspeccionar el comportamiento de un detector concreto sobre 
un vídeo o bundle de debug ya generado, sin necesidad de re-ejecutar 
el pipeline completo.

Ejemplo:

```bash
source .venv/bin/activate
PYTHONPATH=. python -m src.scripts.run_spine_flexion_debug
```

Estos scripts están orientados al desarrollo y validación del sistema. 
No son necesarios para usar la aplicación ni para ejecutar el análisis 
con `data/ejemplo_rdl.mp4`.

## Reproducibilidad

- Entornos, vídeos, pesos y la mayoría de `outputs/` no se versionan.
- Se ignoran extensiones pesadas (`.mp4`, `.pth`, `.pt`, `.npz` en general).
- Documentación del paquete pipeline: `src/pipeline/README.md`.
