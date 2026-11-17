import argparse
import itertools
import sys
import time

import numpy as np
import ray
from scipy.spatial import cKDTree

HEALTHY, INFECTED, DEAD = 0, 1, 2
CHARS = {HEALTHY: "\033[32m.\033[0m", INFECTED: "\033[31m*\033[0m", DEAD: "\033[90mx\033[0m"}


@ray.remote
class Worker:
    def move(self, pos_chunk, alive_chunk, move_speed, width, height, seed):
        pos_chunk = pos_chunk.copy()
        rng = np.random.default_rng(seed)
        delta = rng.uniform(-move_speed, move_speed, size=pos_chunk.shape)
        pos_chunk[alive_chunk] += delta[alive_chunk]
        np.clip(pos_chunk[:, 0], 0, width, out=pos_chunk[:, 0])
        np.clip(pos_chunk[:, 1], 0, height, out=pos_chunk[:, 1])
        return pos_chunk

    def find_hits(self, healthy_pos, infected_pos_chunk, radius):
        node_ip = ray.util.get_node_ip_address()
        if infected_pos_chunk.shape[0] == 0 or healthy_pos.shape[0] == 0:
            return np.empty(0, dtype=np.int64), node_ip
        tree = cKDTree(healthy_pos)
        results = tree.query_ball_point(infected_pos_chunk, r=radius, workers=-1)
        count = sum(len(r) for r in results)
        hits = np.fromiter(itertools.chain.from_iterable(results), dtype=np.int64, count=count)
        return np.unique(hits), node_ip


def spawn_workers():
    workers, weights = [], []
    for n in ray.nodes():
        if not n["Alive"]:
            continue
        ip = n["NodeManagerAddress"]
        cpu = int(n["Resources"].get("CPU", 1))
        workers.append(Worker.options(resources={f"node:{ip}": 0.01}, num_cpus=cpu).remote())
        weights.append(cpu)
    return workers, weights


def parse_args():
    p = argparse.ArgumentParser(description="City virus simulation on Ray (numpy/scipy, built for millions of people)")
    p.add_argument("--width", type=float, default=3000.0)
    p.add_argument("--height", type=float, default=3000.0)
    p.add_argument("--people", type=int, default=1_000_000)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--radius", type=float, default=3.0)
    p.add_argument("--infect-prob", type=float, default=0.3)
    p.add_argument("--death-prob", type=float, default=0.015)
    p.add_argument("--move-speed", type=float, default=5.0)
    p.add_argument("--initial-infected", type=int, default=50)
    p.add_argument("--interval", type=float, default=0.0, help="artificial pause per step in seconds (0 = full speed)")
    p.add_argument("--render-w", type=int, default=70)
    p.add_argument("--render-h", type=int, default=28)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def render(pos, status, args):
    rw, rh = args.render_w, args.render_h
    cx = np.clip((pos[:, 0] / args.width * rw).astype(np.int64), 0, rw - 1)
    cy = np.clip((pos[:, 1] / args.height * rh).astype(np.int64), 0, rh - 1)
    flat = cy * rw + cx

    dead_cells = np.zeros(rw * rh, dtype=bool)
    infected_cells = np.zeros(rw * rh, dtype=bool)
    healthy_cells = np.zeros(rw * rh, dtype=bool)
    dead_cells[flat[status == DEAD]] = True
    infected_cells[flat[status == INFECTED]] = True
    healthy_cells[flat[status == HEALTHY]] = True

    lines = ["+" + "-" * rw + "+"]
    for row in range(rh):
        s = row * rw
        e = s + rw
        chars = np.where(dead_cells[s:e], "D", np.where(infected_cells[s:e], "I", np.where(healthy_cells[s:e], "H", " ")))
        line = "".join(CHARS[{"H": HEALTHY, "I": INFECTED, "D": DEAD}[c]] if c != " " else " " for c in chars)
        lines.append("|" + line + "|")
    lines.append("+" + "-" * rw + "+")
    return "\n".join(lines)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    ray.init(logging_level="ERROR")  # joins a running cluster, otherwise starts Ray locally
    workers, weights = spawn_workers()
    total_weight = sum(weights)
    nodes_seen = set()

    n = args.people
    pos = rng.uniform([0, 0], [args.width, args.height], size=(n, 2))
    status = np.zeros(n, dtype=np.int8)
    status[rng.choice(n, size=min(args.initial_infected, n), replace=False)] = INFECTED

    sys.stdout.write("\033[2J")
    t0 = time.time()
    n_infected = args.initial_infected

    for step in range(1, args.steps + 1):
        step_t0 = time.time()

        alive = status != DEAD
        move_bounds = np.cumsum([0] + [round(n * w / total_weight) for w in weights])
        move_bounds[-1] = n
        move_futures = [
            workers[i].move.remote(
                pos[move_bounds[i]:move_bounds[i + 1]],
                alive[move_bounds[i]:move_bounds[i + 1]],
                args.move_speed, args.width, args.height,
                rng.integers(0, 2**31 - 1),
            )
            for i in range(len(workers))
            if move_bounds[i + 1] > move_bounds[i]
        ]
        moved_chunks = ray.get(move_futures)
        pos = np.concatenate(moved_chunks)

        healthy_idx = np.where(status == HEALTHY)[0]
        infected_idx = np.where(status == INFECTED)[0]
        if healthy_idx.size and infected_idx.size:
            healthy_pos = pos[healthy_idx]
            infected_pos = pos[infected_idx]
            bounds = np.cumsum([0] + [round(infected_idx.size * w / total_weight) for w in weights])
            bounds[-1] = infected_idx.size
            futures = [
                workers[i].find_hits.remote(healthy_pos, infected_pos[bounds[i]:bounds[i + 1]], args.radius)
                for i in range(len(workers))
                if bounds[i + 1] > bounds[i]
            ]
            step_nodes = set()
            all_hits = []
            for local_hits, node_ip in ray.get(futures):
                step_nodes.add(node_ip)
                if local_hits.size:
                    all_hits.append(local_hits)
            nodes_seen |= step_nodes
            if all_hits:
                uniq = np.unique(np.concatenate(all_hits))
                draws = rng.random(uniq.size)
                newly = uniq[draws < args.infect_prob]
                status[healthy_idx[newly]] = INFECTED

        infected_mask = status == INFECTED
        n_inf_now = int(infected_mask.sum())
        if n_inf_now:
            draws = rng.random(n_inf_now)
            die_idx = np.where(infected_mask)[0][draws < args.death_prob]
            status[die_idx] = DEAD

        counts = np.bincount(status, minlength=3)
        n_healthy, n_infected, n_dead = int(counts[0]), int(counts[1]), int(counts[2])

        elapsed = time.time() - t0
        step_dt = time.time() - step_t0
        sps = 1.0 / step_dt if step_dt > 0 else float("inf")
        pps = n / step_dt if step_dt > 0 else float("inf")

        sys.stdout.write("\033[H")
        sys.stdout.write(render(pos, status, args) + "\n")
        sys.stdout.write(
            f"step {step}/{args.steps}  t={elapsed:6.1f}s  "
            f"healthy={n_healthy:8d}  infected={n_infected:8d}  dead={n_dead:8d}  "
            f"({sps:6.1f} steps/s, {pps:14,.0f} people/s)\033[K\n"
        )
        sys.stdout.write(f"Cluster nodes used so far: {', '.join(sorted(nodes_seen)) or '-'}\033[K\n")
        sys.stdout.flush()

        if n_infected == 0:
            break
        if args.interval:
            time.sleep(args.interval)

    print("\nOutbreak over." if n_infected == 0 else "\nReached the step limit.")


if __name__ == "__main__":
    main()
