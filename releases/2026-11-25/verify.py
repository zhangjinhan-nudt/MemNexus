"""Verify frozen source text without network or provider calls."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
failures = []
for name, expected in manifest["files"].items():
    path = ROOT / name
    if not path.is_file():
        failures.append(name + ": missing")
        continue
    data = path.read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(data).hexdigest() != expected["sha256_lf"]:
        failures.append(name + ": changed")
if failures:
    raise SystemExit("Frozen file verification failed:\n" + "\n".join(failures))
print(f"Verified {len(manifest['files'])} frozen files; no network requests.")
