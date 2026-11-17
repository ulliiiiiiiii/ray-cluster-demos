# ray-cluster-demos

Two small programs that put a [Ray](https://www.ray.io/) cluster to work and show it in the terminal.
They run on a single machine as well; on a cluster, Ray spreads the work across all nodes.

## city_virus.py

A virus spreading through a city of up to a million people. In every step, Ray workers move people
around, find who is close to an infected person (a KD-tree from SciPy) and roll infection and death.
The city is drawn as a map of healthy (`.`), infected (`*`) and dead (`x`) people, with live counters
and the nodes that did the work.

```sh
python city_virus.py --people 200000 --steps 500
```

Options: `--width`, `--height`, `--people`, `--steps`, `--radius`, `--infect-prob`, `--death-prob`,
`--move-speed`, `--initial-infected`, `--interval`, `--render-w`, `--render-h`, `--seed`.

## mandelbrot_ray.py

A slow, long Mandelbrot zoom rendered as ASCII art, in color if you like. Rows are computed as Ray
tasks, and a table shows how much each node contributed.

```sh
python mandelbrot_ray.py --preset seahorse --color
```

Presets: `default`, `seahorse`, `elephant`, `spiral`. `--frames` limits the run, `--width` and
`--height` set the canvas, `--max-iter` the detail. `python mandelbrot_ray.py --help` lists the rest.

## Setup

```sh
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

On a single machine, just run the scripts and Ray starts locally. For a cluster, start Ray on the
head node with `ray start --head`, join the other machines with `ray start --address=<head>:6379`,
then run a script on any node: it joins the running cluster.

## License

MIT
