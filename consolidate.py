"""Consolidate the whole codebase into one .txt file.

Run it from anywhere inside the repo:

    python consolidate.py

It writes  memgram_code_<YYYYMMDD_HHMMSS>.txt  to the repo root: every source file,
in a stable order, each preceded by a `===== path =====` header, with an index at
the top. Secrets (.env, .claude), dependencies (node_modules, venvs), binaries
(.docx/.pptx/.pyc), and generated output are skipped, so the result is safe to
share or paste into an LLM for review.
"""
from __future__ import annotations

import datetime
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Directories we never descend into.
SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".next", "out", "dist", "build",
    ".venv", "venv", "env", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".idea", ".vscode", ".claude", "pgdata",
}

# Files we skip outright (secrets + generated + lockfiles).
SKIP_FILES = {".env", ".env.local", "package-lock.json", "yarn.lock",
              "bench/quality/results.json"}

# Only these extensions (plus a few exact names below) are treated as source.
KEEP_EXT = {
    ".py", ".sql", ".ts", ".tsx", ".js", ".mjs", ".jsx", ".json", ".yml",
    ".yaml", ".toml", ".md", ".css", ".txt", ".sh", ".cfg", ".ini", ".example",
}
KEEP_NAMES = {"Dockerfile", ".gitignore", ".dockerignore", ".env.example",
              "postgres-age.Dockerfile"}

# Never include binaries / docs even if the walk reaches them.
SKIP_EXT = {".pyc", ".pyo", ".docx", ".pptx", ".pdf", ".png", ".jpg", ".jpeg",
            ".gif", ".ico", ".woff", ".woff2", ".ttf", ".eot", ".zip", ".exe",
            ".dll", ".so", ".sqlite"}


def wanted(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    if rel in SKIP_FILES:
        return False
    if path.name in SKIP_FILES:
        return False
    if path.suffix.lower() in SKIP_EXT:
        return False
    if path.name.startswith("memgram_code_") and path.suffix == ".txt":
        return False  # don't include a previous consolidation
    return path.suffix.lower() in KEEP_EXT or path.name in KEEP_NAMES


def collect() -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            p = Path(dirpath) / name
            if wanted(p):
                files.append(p)
    return files


def main() -> None:
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = ROOT / f"memgram_code_{stamp}.txt"
    files = collect()

    with out_path.open("w", encoding="utf-8") as out:
        out.write("=" * 80 + "\n")
        out.write(f"MEMGRAM — consolidated source\n")
        out.write(f"generated: {datetime.datetime.now().isoformat(timespec='seconds')}\n")
        out.write(f"files: {len(files)}\n")
        out.write("=" * 80 + "\n\n")

        out.write("INDEX\n")
        for p in files:
            out.write(f"  {p.relative_to(ROOT).as_posix()}\n")
        out.write("\n")

        total_lines = 0
        for p in files:
            rel = p.relative_to(ROOT).as_posix()
            try:
                text = p.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue  # skip anything that isn't clean text
            total_lines += text.count("\n") + 1
            out.write("\n" + "=" * 80 + "\n")
            out.write(f"FILE: {rel}\n")
            out.write("=" * 80 + "\n")
            out.write(text)
            if not text.endswith("\n"):
                out.write("\n")

    size_kb = out_path.stat().st_size / 1024
    print(f"Wrote {out_path.name}  ({len(files)} files, ~{total_lines} lines, {size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
