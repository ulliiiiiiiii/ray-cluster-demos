import argparse
import math
import os
import sys
import time
from dataclasses import dataclass

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import ray


ESCAPE_CHARS = " .,:;irsXA253hMHGS#9B&@"

PRESETS = {
    "default": {
        "center_x": -0.743643887037151,
        "center_y": 0.13182590420533,
        "initial_scale": 3.0,
        "zoom_step": 0.985,
        "max_iter": 250,
        "drift_points": [
            (-0.743643887037151, 0.13182590420533),
            (-0.748, 0.1),
            (-0.1, 0.651),
            (-1.25066, 0.02012),
        ],
    },
    "seahorse": {
        "center_x": -0.74543,
        "center_y": 0.11301,
        "initial_scale": 0.9,
        "zoom_step": 0.989,
        "max_iter": 350,
        "drift_points": [
            (-0.74543, 0.11301),
            (-0.74497, 0.114),
            (-0.7447, 0.1118),
            (-0.7464, 0.1105),
        ],
    },
    "elephant": {
        "center_x": 0.285,
        "center_y": 0.01,
        "initial_scale": 2.4,
        "zoom_step": 0.987,
        "max_iter": 220,
        "drift_points": [
            (0.285, 0.01),
            (0.27, 0.015),
            (0.3, -0.02),
            (0.28, 0.04),
        ],
    },
    "spiral": {
        "center_x": -0.1011,
        "center_y": 0.9563,
        "initial_scale": 0.8,
        "zoom_step": 0.992,
        "max_iter": 320,
        "drift_points": [
            (-0.1011, 0.9563),
            (-0.1015, 0.9552),
            (-0.1003, 0.9571),
            (-0.1022, 0.9548),
        ],
    },
}


@dataclass
class NodeState:
    ip: str
    cpu: float
    slots: int
    inflight: int = 0
    submitted_batches: int = 0
    completed_batches: int = 0
    rows_done: int = 0
    last_rows: int = 0
    last_elapsed: float = 0.0


def parse_args():
    parser = argparse.ArgumentParser(description="Mandelbrot zoom rendered in the terminal, computed on Ray")
    parser.add_argument("--preset", choices=sorted(PRESETS.keys()), default="default")
    parser.add_argument("--width", type=int, default=120)
    parser.add_argument("--height", type=int, default=60)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--frames", type=int, default=10_000)
    parser.add_argument("--batch-rows", type=int, default=6)
    parser.add_argument("--task-cpu", type=float, default=0.55)
    parser.add_argument("--task-memory-mb", type=int, default=128)
    parser.add_argument("--duty-cycle", type=float, default=0.45)
    parser.add_argument("--pause", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--center-x", type=float, default=None)
    parser.add_argument("--center-y", type=float, default=None)
    parser.add_argument("--zoom-step", type=float, default=None)
    parser.add_argument("--initial-scale", type=float, default=None)
    parser.add_argument("--row-sleep", type=float, default=0.0, help="Extra sleep per rendered row chunk")
    parser.add_argument("--drift-strength", type=float, default=0.18, help="how strongly the loop drifts towards the preset points")
    parser.add_argument("--drift-window", type=int, default=80, help="how many frames per drift section")
    parser.add_argument("--color", dest="color", action="store_true", default=True)
    parser.add_argument("--no-color", dest="color", action="store_false")
    return parser.parse_args()


def resolve_config(args):
    preset = PRESETS[args.preset]
    return {
        "preset": args.preset,
        "center_x": preset["center_x"] if args.center_x is None else args.center_x,
        "center_y": preset["center_y"] if args.center_y is None else args.center_y,
        "initial_scale": preset["initial_scale"] if args.initial_scale is None else args.initial_scale,
        "zoom_step": preset["zoom_step"] if args.zoom_step is None else args.zoom_step,
        "max_iter": preset["max_iter"] if args.max_iter is None else args.max_iter,
        "drift_points": preset["drift_points"],
    }


def discover_nodes(task_cpu: float) -> list[NodeState]:
    nodes: list[NodeState] = []
    for node in ray.nodes():
        if not node.get("Alive"):
            continue
        ip = node["NodeManagerAddress"]
        cpu = float(node["Resources"].get("CPU", 1.0))
        slots = max(1, int(cpu / max(task_cpu, 0.01)))
        nodes.append(NodeState(ip=ip, cpu=cpu, slots=slots))

    if not nodes:
        raise RuntimeError("No active Ray nodes found.")

    return nodes


def palette(value: float) -> str:
    idx = min(len(ESCAPE_CHARS) - 1, max(0, int(value * (len(ESCAPE_CHARS) - 1))))
    return ESCAPE_CHARS[idx]


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def smoothstep(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def color_for(value: float):
    value = min(max(value, 0.0), 1.0)
    if value < 0.33:
        t = value / 0.33
        return int(20 + 20 * t), int(40 + 40 * t), int(100 + 120 * t)
    if value < 0.66:
        t = (value - 0.33) / 0.33
        return int(40 + 180 * t), int(80 + 120 * t), int(220 - 120 * t)
    t = (value - 0.66) / 0.34
    return int(220 + 35 * t), int(220 + 30 * t), int(120 + 120 * t)


def style_char(ch: str, value: float, color_enabled: bool) -> str:
    if not color_enabled or ch == " ":
        return ch
    r, g, b = color_for(value)
    return f"\033[38;2;{r};{g};{b}m{ch}\033[0m"


@ray.remote
def render_rows(
    y_start: int,
    y_stop: int,
    width: int,
    height: int,
    center_x: float,
    center_y: float,
    scale: float,
    max_iter: int,
    duty_cycle: float,
    row_sleep: float,
    color_enabled: bool,
):
    t0 = time.perf_counter()
    rows = []
    duty_cycle = min(max(duty_cycle, 0.01), 1.0)
    row_sleep = max(0.0, row_sleep)

    span_x = scale
    span_y = scale * (height / width)

    for y in range(y_start, y_stop):
        row_t0 = time.perf_counter()
        imag = center_y + (y - height / 2.0) * (span_y / height)
        line = []
        for x in range(width):
            real = center_x + (x - width / 2.0) * (span_x / width)
            c = complex(real, imag)
            z = 0.0j
            n = 0
            while (z.real * z.real + z.imag * z.imag) <= 4.0 and n < max_iter:
                z = z * z + c
                n += 1
            if n >= max_iter:
                line.append(style_char("@", 1.0, color_enabled))
            else:
                smooth = n - math.log(math.log(abs(z)) + 1e-12, 2)
                value = min(max(smooth / max_iter, 0.0), 1.0)
                line.append(style_char(palette(value), value, color_enabled))
        rows.append("".join(line))

        if duty_cycle < 1.0:
            work = time.perf_counter() - row_t0
            extra = work * (1.0 - duty_cycle) / duty_cycle + row_sleep
            if extra > 0:
                time.sleep(extra)

    elapsed = time.perf_counter() - t0
    return y_start, y_stop, rows, elapsed, ray.util.get_node_ip_address()


def render_dashboard(
    nodes: list[NodeState],
    frame_idx: int,
    total_frames: int,
    scale: float,
    center_x: float,
    center_y: float,
    t0: float,
    rows_done: int,
    width: int,
    height: int,
    max_iter: int,
    preset_name: str,
    color_enabled: bool,
    drift_strength: float,
):
    elapsed = max(time.perf_counter() - t0, 1e-9)
    rows_per_sec = rows_done / elapsed
    frames_left = max(0, total_frames - frame_idx)
    eta = frames_left * (height / max(rows_per_sec, 1e-9))

    lines = []
    lines.append("\033[H\033[2J")
    lines.append("Ray Mandelbrot Zoom")
    lines.append(
        f"Frame {frame_idx}/{total_frames}  "
        f"Scale={scale:.8f}  "
        f"Center=({center_x:.12f}, {center_y:.12f})  "
        f"Rows/s={rows_per_sec:,.0f}  ETA={eta:,.1f}s"
    )
    lines.append(f"Preset={preset_name}  Color={'on' if color_enabled else 'off'}  Drift={drift_strength:.2f}")
    lines.append("")
    lines.append(
        "Node".ljust(16)
        + " CPU ".rjust(6)
        + " Slots ".rjust(7)
        + " Inf ".rjust(5)
        + " Sent ".rjust(7)
        + " Done ".rjust(7)
        + " Rows ".rjust(12)
        + " LastRows ".rjust(10)
        + " LastTime ".rjust(12)
    )

    for node in nodes:
        lines.append(
            f"{node.ip:<16}"
            f"{node.cpu:6.1f}"
            f"{node.slots:7d}"
            f"{node.inflight:5d}"
            f"{node.submitted_batches:7d}"
            f"{node.completed_batches:7d}"
            f"{node.rows_done:12,d}"
            f"{node.last_rows:10,d}"
            f"{node.last_elapsed:12.2f}s"
        )

    lines.append("")
    lines.append("Tip: every finished frame is a clean terminal picture, good for screen recordings.")
    lines.append(f"Canvas: {width}x{height}  MaxIter={max_iter}")

    sys.stdout.write("\033[?25l")
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()


def build_schedule(nodes: list[NodeState], height: int, batch_rows: int):
    jobs = []
    y = 0
    while y < height:
        y_stop = min(height, y + batch_rows)
        jobs.append((y, y_stop))
        y = y_stop

    schedule = []
    idx = 0
    while idx < len(jobs):
        for node in nodes:
            for _ in range(node.slots):
                if idx >= len(jobs):
                    break
                schedule.append((node, jobs[idx][0], jobs[idx][1]))
                idx += 1
    return schedule


def frame_center(config, args, frame_idx: int, phase_offset: int):
    zoom = config["initial_scale"] * (config["zoom_step"] ** frame_idx)
    points = config["drift_points"]
    window = max(1, args.drift_window)

    if points and args.drift_strength > 0:
        seg = (phase_offset + frame_idx // window) % len(points)
        nxt = (seg + 1) % len(points)
        local = smoothstep((frame_idx % window) / window)
        drift_x = lerp(points[seg][0], points[nxt][0], local)
        drift_y = lerp(points[seg][1], points[nxt][1], local)
        center_x = lerp(config["center_x"], drift_x, args.drift_strength)
        center_y = lerp(config["center_y"], drift_y, args.drift_strength)
    else:
        center_x = config["center_x"]
        center_y = config["center_y"]

    return zoom, center_x, center_y


def main():
    args = parse_args()
    config = resolve_config(args)
    rng = np.random.default_rng(args.seed)
    phase_offset = int(rng.integers(0, max(1, len(config["drift_points"]))))

    ray.init(logging_level="ERROR")  # joins a running cluster, otherwise starts Ray locally
    nodes = discover_nodes(args.task_cpu)
    schedule = build_schedule(nodes, args.height, args.batch_rows)

    try:
        for frame_idx in range(1, args.frames + 1):
            scale, center_x, center_y = frame_center(config, args, frame_idx - 1, phase_offset)

            for node in nodes:
                node.inflight = 0
                node.submitted_batches = 0
                node.completed_batches = 0
                node.rows_done = 0
                node.last_rows = 0
                node.last_elapsed = 0.0

            pending: dict[object, tuple[NodeState, int, int]] = {}
            frame_rows = [" " * args.width for _ in range(args.height)]
            rows_done = 0
            t0 = time.perf_counter()

            for node, y_start, y_stop in schedule:
                fut = render_rows.options(
                    resources={f"node:{node.ip}": 0.01},
                    num_cpus=args.task_cpu,
                    memory=args.task_memory_mb * 1024 * 1024,
                ).remote(
                    y_start,
                    y_stop,
                    args.width,
                    args.height,
                    center_x,
                    center_y,
                    scale,
                    config["max_iter"],
                    args.duty_cycle,
                    args.row_sleep,
                    args.color,
                )
                node.inflight += 1
                node.submitted_batches += 1
                pending[fut] = (node, y_start, y_stop)

            render_dashboard(
                nodes,
                frame_idx,
                args.frames,
                scale,
                center_x,
                center_y,
                t0,
                rows_done,
                args.width,
                args.height,
                config["max_iter"],
                config["preset"],
                args.color,
                args.drift_strength,
            )

            while pending:
                ready, _ = ray.wait(list(pending.keys()), num_returns=1)
                fut = ready[0]
                node, _, _ = pending.pop(fut)
                got_y_start, got_y_stop, rows, elapsed, ip = ray.get(fut)
                if ip != node.ip:
                    node.ip = ip
                node.inflight -= 1
                node.completed_batches += 1
                node.rows_done += got_y_stop - got_y_start
                node.last_rows = got_y_stop - got_y_start
                node.last_elapsed = elapsed
                for i, row in enumerate(rows, start=got_y_start):
                    frame_rows[i] = row
                rows_done += got_y_stop - got_y_start
                render_dashboard(
                    nodes,
                    frame_idx,
                    args.frames,
                    scale,
                    center_x,
                    center_y,
                    t0,
                    rows_done,
                    args.width,
                    args.height,
                    config["max_iter"],
                    config["preset"],
                    args.color,
                    args.drift_strength,
                )
                if args.pause:
                    time.sleep(args.pause)

            sys.stdout.write("\033[H\033[2J")
            sys.stdout.write(f"Ray Mandelbrot Frame {frame_idx}/{args.frames}\n")
            sys.stdout.write(
                f"Preset={config['preset']}  Scale={scale:.8f}  "
                f"Center=({center_x:.12f}, {center_y:.12f})  Workers={len(nodes)}\n\n"
            )
            for row in frame_rows:
                sys.stdout.write(row + "\n")
            sys.stdout.flush()

            if args.pause:
                time.sleep(args.pause)

    finally:
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
