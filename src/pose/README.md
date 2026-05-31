# `pose`

Convierte videos en keypoints.

- `detection.py`: detecta personas en cada frame.
- `selection.py`: decide que persona usar si hay varias.
- `model.py`: wrapper del modelo RTMPose.
- `extraction.py`: flujo reutilizable video -> `.npz`.

No debe contener segmentacion, comparacion biomecanica ni feedback.

