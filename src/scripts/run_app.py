
from __future__ import annotations

import os

import flet as ft

from src.app.flet_app import main


if __name__ == "__main__":
    host = os.getenv("APP_HOST", "127.0.0.1")
    port = int(os.getenv("APP_PORT", "8550"))

    print(f"Interfaz disponible en: http://{host}:{port}")

    # La API de Flet varía según versión.
    try:
        ft.run(
            main,
            view=ft.AppView.WEB_BROWSER,
            host=host,
            port=port,
        )
    except TypeError:
        ft.app(
            target=main,
            view=ft.AppView.WEB_BROWSER,
            host=host,
            port=port,
        )

