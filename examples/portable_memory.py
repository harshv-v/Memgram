"""Memory on a pendrive — export your entire memory to a file, carry it, load it
into any Memgram instance.

    # on machine A: dump everything for a user/project to a portable file
    python examples/portable_memory.py export --project demo --user u1 --out memory.json

    # copy memory.json to a USB stick, move to machine B, then:
    python examples/portable_memory.py import --file memory.json --project demo --user u1

The bundle is plain JSON (content + metadata, no instance-specific vectors). On
import the target instance RE-EMBEDS, so it works across embedder/dimension choices.
You own your memory; it travels with you.
"""
import argparse
import json
import os
import sys

import httpx

API = os.environ.get("MEMGRAM_API_BASE_URL", "http://localhost:8000")
KEY = os.environ.get("MEMGRAM_API_KEY", "mgram_dev_key")
H = {"Authorization": f"Bearer {KEY}"}


def _counts(bundle: dict) -> str:
    return ", ".join(f"{k}={len(bundle.get(k, []))}" for k in
                     ("instructions", "semantic_memories", "episodic_logs"))


def do_export(args):
    c = httpx.Client(base_url=API, headers=H, timeout=60)
    r = c.get(f"/v1/memory/export/{args.user}", params={"project_id": args.project})
    r.raise_for_status()
    bundle = r.json()
    # drop instance-specific vectors — the target re-embeds on import
    for m in bundle.get("semantic_memories", []):
        m.pop("embedding", None)
    payload = {"memgram_export_version": 1,
               "source": {"project": args.project, "user": args.user},
               "data": bundle}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"Exported {_counts(bundle)} -> {args.out}")


def do_import(args):
    with open(args.file, encoding="utf-8") as f:
        payload = json.load(f)
    data = payload.get("data", payload)
    c = httpx.Client(base_url=API, headers=H, timeout=120)
    r = c.post("/v1/memory/import", json={
        "data": data,
        "target_project_id": args.project, "target_user_id": args.user})
    r.raise_for_status()
    print(f"Imported {r.json()['imported']} into project={args.project} user={args.user}")


def main():
    p = argparse.ArgumentParser(description="Portable Memgram memory (export/import).")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export"); e.add_argument("--project", required=True)
    e.add_argument("--user", required=True); e.add_argument("--out", default="memory.json")
    e.set_defaults(func=do_export)
    i = sub.add_parser("import"); i.add_argument("--file", required=True)
    i.add_argument("--project", required=True); i.add_argument("--user", required=True)
    i.set_defaults(func=do_import)
    args = p.parse_args()
    try:
        args.func(args)
    except httpx.HTTPStatusError as ex:
        sys.exit(f"API error: {ex.response.status_code} {ex.response.text[:200]}")


if __name__ == "__main__":
    main()
