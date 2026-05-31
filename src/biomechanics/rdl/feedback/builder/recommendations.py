
from __future__ import annotations

from typing import Any


_RECOMMENDATIONS: dict[str, str] = {
    "neck_extension_up": "Durante la bajada, manten la cabeza alineada con el torso. Evita mirar al frente o hacia arriba; piensa en llevar la mirada ligeramente hacia el suelo, acompanando la inclinacion del tronco.",
    "neck_flexion_down": "Manten el cuello neutro y evita dejar caer la cabeza. Piensa en alargar la nuca y mirar a un punto del suelo unos metros por delante.",
    "hip_hinge": "Inicia la bajada llevando la cadera hacia atras, no doblando solo el tronco. Manten una ligera flexion de rodilla y busca sentir tension progresiva en isquios.",
    "knee_dominant": "Reduce la flexion de rodilla durante la bajada. Piensa en desplazar la cadera hacia atras y mantener las espinillas relativamente verticales.",
    "bar_far": "Manten la barra o las manos cerca del cuerpo durante todo el recorrido. Piensa en rozar las piernas y dejar que la barra baje pegada a muslos y tibias.",
    "bent_arms": "Manten los codos extendidos y los brazos relajados. Piensa en que los brazos solo cuelgan de la barra, sin tirar con biceps.",
    "lockout": "Termina cada repeticion recuperando la posicion alta: cadera extendida, torso erguido y controlado, sin hiperextender la espalda.",
    "short_rom": "Permite que la cadera viaje hacia atras hasta alcanzar un rango suficiente, manteniendo tension en isquios y columna neutra. No recortes la bajada salvo que pierdas la posicion.",
    "asymmetry": "Busca repartir el peso de forma equilibrada entre ambos pies. Revisa la posicion inicial, el agarre y que la barra baje centrada.",
    "asymmetry_arms": "Comprueba que ambos brazos cuelgan igual y que no tiras mas con un lado. Manten hombros y manos equilibrados.",
    "asymmetry_legs": "Revisa la colocacion de los pies y la presion contra el suelo. Intenta empujar de forma simetrica con ambas piernas.",
    "spine_flexion": "Reduce el rango si pierdes la posicion del tronco. Antes de bajar mas, fija la caja toracica, manten tension abdominal y lleva la cadera atras sin colapsar el torso.",
    "spine_flexion_possible": (
        "Corrige primero el error principal identificado en esa zona de la bajada y vuelve a grabar. "
        "Practica la activacion del core antes de iniciar el descenso. "
        "Manten la espalda en posicion neutra durante toda la bajada."
    ),
    "spine_flexion_no_hip_failure": (
        "Trabaja la activacion del core y la rigidez del tronco antes de iniciar el descenso. "
        "Asegurate de empujar las caderas hacia atras activamente al bajar para que el tronco no caiga por delante."
    ),
    "neck_movement": "Manten una posicion cervical neutra durante toda la repeticion. Acompana la inclinacion del tronco sin mover en exceso la cabeza hacia arriba o abajo.",
}


def get_recommendation(error_code: str, issue: dict[str, Any]) -> str:
    _ = issue
    code = str(error_code or "")
    return _RECOMMENDATIONS.get(
        code,
        "Revisa este punto de la tecnica y repite el movimiento de forma mas controlada, priorizando estabilidad y rango util.",
    )
