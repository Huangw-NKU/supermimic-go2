"""Record CPU/GPU resources for a live Go2 run without modifying its training state."""

import argparse
import csv
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import psutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--interval", type=float, default=5)
    args = parser.parse_args()
    if args.interval < 1:
        parser.error("interval must be at least one second")
    directory = Path(args.run_dir)
    for _ in range(120):
        if (directory / "process.json").exists():
            break
        time.sleep(1)
    metadata = json.loads((directory / "process.json").read_text())
    process = psutil.Process(metadata["pid"])
    if "supermimic.go2.train" not in process.cmdline():
        raise RuntimeError("process.json no longer identifies the Go2 training process")
    process.cpu_percent()
    psutil.cpu_percent()
    fields = [
        "utc_time",
        "pid",
        "process_cpu_percent",
        "host_cpu_percent",
        "rss_gib",
        "gpu_utilization_percent",
        "gpu_memory_used_mib",
        "gpu_power_w",
        "gpu_temperature_c",
        "training_process_gpu_mib",
    ]
    with (directory / "resources.csv").open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if stream.tell() == 0:
            writer.writeheader()
        while True:
            try:
                if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
                    break
                time.sleep(args.interval)
                gpu = (
                    subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu",
                            "--format=csv,noheader,nounits",
                        ],
                        text=True,
                        timeout=5,
                    )
                    .splitlines()[0]
                    .split(",")
                )
                processes = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=5,
                ).splitlines()
                memory = next(
                    (
                        float(line.split(",")[1])
                        for line in processes
                        if int(line.split(",")[0]) == process.pid
                    ),
                    0.0,
                )
                row = dict(
                    zip(
                        fields,
                        [
                            datetime.now(UTC).isoformat(),
                            process.pid,
                            process.cpu_percent(),
                            psutil.cpu_percent(),
                            process.memory_info().rss / 2**30,
                            *[float(value) for value in gpu],
                            memory,
                        ],
                        strict=True,
                    )
                )
                writer.writerow(row)
                stream.flush()
                temporary = directory / "resource_status.tmp"
                temporary.write_text(
                    json.dumps({"running": True, **row}, indent=2) + "\n"
                )
                temporary.replace(directory / "resource_status.json")
            except psutil.NoSuchProcess:
                break
            except (subprocess.SubprocessError, ValueError) as error:
                print(f"Resource sample unavailable: {error}", flush=True)
    (directory / "resource_status.json").write_text(
        json.dumps(
            {
                "running": False,
                "pid": process.pid,
                "utc_time": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
