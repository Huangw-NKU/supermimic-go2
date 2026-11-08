"""Launch a documented Go2 configuration; --dry-run prints the command only."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    presets = sorted(p.stem for p in (ROOT / "configs").glob("*.json"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preset", choices=presets)
    parser.add_argument("run_name")
    parser.add_argument("--dry-run", action="store_true")
    args, overrides = parser.parse_known_args()
    settings = json.loads((ROOT / "configs" / f"{args.preset}.json").read_text())
    command = [str(ROOT / "run_go2_train.sh"), args.run_name]
    for key, value in settings.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            command.append(flag if value else f"--no-{key}")
        elif isinstance(value, list):
            command.extend([flag, *map(str, value)])
        else:
            command.extend([flag, str(value)])
    command.extend(overrides)
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
