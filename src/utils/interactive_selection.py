
from __future__ import annotations

from pathlib import Path


def ask_source_baseline_tests(*, tests_available: bool, entity_label: str) -> str:
    if not tests_available:
        print(f"No existe carpeta tests para {entity_label}; usando baseline.")
        return "baseline"

    print(f"Selecciona origen de {entity_label}:\n")
    print("1. baseline")
    print("2. tests")
    raw = input("\nIntroduce el número: ").strip()
    if raw == "1":
        return "baseline"
    if raw == "2":
        return "tests"
    raise SystemExit("Entrada inválida: selecciona 1 o 2.")


def browse_files_interactively(
    root: Path,
    *,
    file_suffixes: tuple[str, ...],
    item_label_singular: str,
    blocked_top_level_dirs: set[str] | None = None,
) -> Path:
    if not root.is_dir():
        raise SystemExit(f"No existe el directorio: {root.as_posix()}")

    blocked_top_level_dirs = blocked_top_level_dirs or set()
    current = root

    while True:
        dirs = sorted(
            p
            for p in current.iterdir()
            if p.is_dir()
            and not (
                current == root and p.name in blocked_top_level_dirs
            )
        )
        files = sorted(
            p for p in current.iterdir() if p.is_file() and p.suffix.lower() in file_suffixes
        )

        rel_current = "." if current == root else current.relative_to(root).as_posix()
        print(f"\nDirectorio actual: {rel_current}")
        if not dirs and not files:
            print("  (vacío)")
            if current == root:
                raise SystemExit(f"No hay {item_label_singular}s disponibles en {root.as_posix()}")
        else:
            for i, d in enumerate(dirs, start=1):
                print(f"{i}. [DIR] {d.name}/")
            offset = len(dirs)
            for j, f in enumerate(files, start=1):
                print(f"{offset + j}. [FILE] {f.name}")

        if current != root:
            print("u. Subir carpeta")

        raw = input(f"\nSelecciona carpeta/{item_label_singular} por número: ").strip().lower()
        if raw == "u":
            if current == root:
                print("Ya estás en la raíz.")
                continue
            current = current.parent
            continue

        if not raw.isdigit():
            print("Entrada inválida: usa un número o 'u'.")
            continue

        idx = int(raw)
        total = len(dirs) + len(files)
        if idx < 1 or idx > total:
            print(f"Número fuera de rango (1..{total}).")
            continue

        if idx <= len(dirs):
            current = dirs[idx - 1]
            continue

        file_idx = idx - len(dirs) - 1
        return files[file_idx]
