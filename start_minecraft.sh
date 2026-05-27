#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "$REPO_ROOT/env.sh" ]]; then
    . "$REPO_ROOT/env.sh"
elif [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
    . "$REPO_ROOT/.venv/bin/activate"
fi

if ! command -v python >/dev/null 2>&1; then
    echo "python not found after activating the repo environment" >&2
    exit 1
fi

# Default to the three clients needed for a two-player video rollout: two players
# plus the spectator client used by record_video=True.
replaceable=0
ports=()
for arg in "$@"; do
    if [[ "$arg" == "--replaceable" ]]; then
        replaceable=1
    else
        ports+=("$arg")
    fi
done

if [[ ${#ports[@]} -eq 0 ]]; then
    ports=(10000 10001 10002)
fi

for port in "${ports[@]}"; do
    if ! [[ "$port" =~ ^[0-9]+$ ]]; then
        echo "all arguments must be numeric Minecraft client ports" >&2
        exit 1
    fi
done

python - "$replaceable" "${ports[@]}" <<'PY'
import importlib.util
import json
import sys
import urllib.request
from pathlib import Path

import malmo

replaceable = bool(int(sys.argv[1]))
ports = [int(port) for port in sys.argv[2:]]

index_path = Path.home() / ".gradle/caches/minecraft/assets/indexes/1.11.json"
objects_dir = Path.home() / ".gradle/caches/minecraft/assets/objects"

if index_path.exists():
    objects = json.loads(index_path.read_text())["objects"]
    missing_hashes = sorted(
        {
            meta["hash"]
            for meta in objects.values()
            if not (objects_dir / meta["hash"][:2] / meta["hash"]).exists()
        }
    )
    if missing_hashes:
        print(
            f"Downloading {len(missing_hashes)} missing Minecraft assets over HTTPS..."
        )
        for index, asset_hash in enumerate(missing_hashes, 1):
            target = objects_dir / asset_hash[:2] / asset_hash
            target.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(
                f"https://resources.download.minecraft.net/{asset_hash[:2]}/{asset_hash}",
                target,
            )
            if index % 50 == 0 or index == len(missing_hashes):
                print(f"{index}/{len(missing_hashes)}")

minecraft_dir = Path(malmo.__file__).resolve().parent / "Minecraft"
launcher_path = minecraft_dir / "launch_minecraft_in_background.py"
spec = importlib.util.spec_from_file_location(
    "launch_minecraft_in_background", launcher_path
)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load launcher from {launcher_path}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

module.launch_minecraft_in_background(
    str(minecraft_dir), ports, timeout=300, replaceable=replaceable, score=False
)
print("Launch requested for ports:", ", ".join(str(port) for port in ports))
PY