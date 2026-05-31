
from __future__ import annotations


_MESSAGES: dict[str, dict[str, str]] = {
    "bent_arms": {
        "title": "Brazos flexionados",
        "what_happens": "Durante la repeticion, los codos no se mantienen completamente extendidos.",
        "why_it_matters": "En el RDL los brazos deberian actuar como correas. Flexionarlos puede alterar la trayectoria de la barra y anadir tension innecesaria.",
    },
    "asymmetry": {
        "title": "Asimetria durante el movimiento",
        "what_happens": "Se observa una diferencia relevante entre ambos lados del cuerpo durante la ejecucion.",
        "why_it_matters": "Una asimetria persistente puede indicar compensaciones, perdida de estabilidad o reparto irregular de la carga.",
    },
    "asymmetry_arms": {
        "title": "Asimetria en brazos",
        "what_happens": "Los brazos no se comportan de forma simetrica durante la repeticion.",
        "why_it_matters": "Esto puede alterar la trayectoria de la barra y generar una ejecucion menos estable.",
    },
    "asymmetry_legs": {
        "title": "Asimetria en piernas",
        "what_happens": "Las piernas no se comportan de forma simetrica durante la repeticion.",
        "why_it_matters": "Esto puede indicar reparto irregular de carga o compensacion entre lados.",
    },
    "bar_far": {
        "title": "Barra alejada del cuerpo",
        "what_happens": "La barra o las manos se alejan demasiado del cuerpo durante el movimiento.",
        "why_it_matters": "Cuanto mas se aleja la barra, mayor es el brazo de palanca y mas dificil es mantener una trayectoria eficiente y estable.",
    },
    "hip_hinge": {
        "title": "Bisagra de cadera insuficiente",
        "what_happens": "Durante la bajada, no llevas la cadera suficientemente hacia atras respecto a la referencia.",
        "why_it_matters": "El peso muerto rumano debe ser un patron dominante de cadera. Si la bisagra es insuficiente, el estimulo sobre la cadena posterior se reduce y la tecnica se acerca a un patron menos eficiente.",
    },
    "knee_dominant": {
        "title": "Exceso de flexion de rodilla",
        "what_happens": "Durante la bajada, flexionas demasiado la rodilla para un peso muerto rumano.",
        "why_it_matters": "Esto desplaza el movimiento hacia un patron mas parecido a una sentadilla parcial y reduce el enfasis sobre la bisagra de cadera.",
    },
    "lockout": {
        "title": "Cierre incompleto arriba",
        "what_happens": "Al finalizar la repeticion, no recuperas completamente la posicion alta.",
        "why_it_matters": "Un cierre incompleto puede indicar falta de control final o una repeticion no terminada correctamente.",
    },
    "neck_movement": {
        "title": "Movimiento cervical no deseado",
        "what_happens": "Durante la ejecucion, el cuello se desvía de una posicion neutra respecto al tronco.",
        "why_it_matters": "Perder alineacion cervical puede reducir la estabilidad del patron y empeorar la calidad tecnica global.",
    },
    "neck_flexion_down": {
        "title": "Cuello flexionado hacia abajo",
        "what_happens": "Durante la ejecucion, la cabeza cae demasiado hacia abajo respecto al tronco.",
        "why_it_matters": "Esto puede romper la alineacion cervical y hacer que el movimiento sea menos estable.",
    },
    "neck_extension_up": {
        "title": "Cuello extendido hacia arriba",
        "what_happens": "Durante la ejecucion, la cabeza se orienta demasiado hacia arriba respecto al tronco.",
        "why_it_matters": "Esto rompe la alineacion cabeza-tronco y puede alterar la posicion cervical durante el peso muerto rumano.",
    },
    "short_rom": {
        "title": "Rango de movimiento insuficiente",
        "what_happens": "No alcanzas suficiente recorrido durante la bajada en comparacion con la referencia.",
        "why_it_matters": "Un rango reducido puede limitar el estimulo tecnico y muscular del ejercicio, especialmente sobre isquios y gluteos.",
    },
    "spine_flexion": {
        "title": "Posible pérdida de posición del tronco",
        "what_happens": (
            "Se observa un patrón compatible con pérdida de posición del tronco durante la bajada. "
            "Esta evaluación se basa en la posición relativa del tronco respecto a la referencia y no en "
            "una medición directa de la columna vertebral."
        ),
        "why_it_matters": "Con una cámara lateral 2D no se puede confirmar flexión real de columna, pero este patrón puede indicar que estás perdiendo la posición del tronco.",
    },
    "spine_flexion_possible": {
        "title": "Posible pérdida de control del tronco",
        "what_happens": (
            "El tronco desciende más de lo esperado en varios tramos de la bajada, pero hay otro error activo "
            "en la misma zona que impide confirmarlo. Una vez corregido ese error y vuelto a analizar, el sistema "
            "podrá evaluar el tronco con más precisión."
        ),
        "why_it_matters": "La flexión no controlada del tronco durante el peso muerto puede aumentar la carga sobre la zona lumbar.",
    },
    "spine_flexion_no_hip_failure": {
        "title": "Tronco más bajo de lo esperado sin causa de bisagra",
        "what_happens": (
            "El tronco desciende por debajo del patrón esperado sin que haya un problema de bisagra de cadera "
            "que lo justifique. Esto puede indicar una pérdida de tensión o control en la espalda durante la bajada."
        ),
        "why_it_matters": "La flexión no controlada del tronco durante el peso muerto puede aumentar la carga sobre la zona lumbar.",
    },
}


def get_message(error_code: str) -> dict[str, str]:
    code = str(error_code or "")
    if code in _MESSAGES:
        return dict(_MESSAGES[code])
    return {
        "title": code if code else "error_tecnico",
        "what_happens": "Se ha detectado una desviacion tecnica asociada a este patron.",
        "why_it_matters": "Conviene revisar este punto porque puede afectar a la calidad de la ejecucion.",
    }
