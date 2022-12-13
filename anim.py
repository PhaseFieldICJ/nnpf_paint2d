#!/usr/bin/env python3

# Parser for the domain bounds
import re
def bounds_parser(s):
    if s is None:
        return None

    bounds = []
    per_dim = s.split('x')
    for dim_spec in per_dim:
        match = re.fullmatch(r'\s*\[([^\]]*)\]\s*', dim_spec)
        if not match:
            raise ValueError(f"Invalid bound specification {dim_spec}")
        bounds.append([float(b) for b in match.group(1).split(',')])
    return bounds

# Command-line arguments
import argparse
from distutils.util import strtobool
parser = argparse.ArgumentParser(
    description="Interactive session for Steiner 2D",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("checkpoint", type=str, help="Path to the model's checkpoint")
parser.add_argument('--bounds', type=bounds_parser, default=None, help="Domain bounds in format like '[0, 1]x[1, 2.5]' (default is model bounds)")
parser.add_argument("--gpu", action="store_true", help="Evaluation model on your GPU")
parser.add_argument("--display_step", type=int, default=1, help="Render frame every given number")
parser.add_argument("--display_infos", type=lambda v: bool(strtobool(v)), nargs='?', default=False, const=True, help="Display simulation and performance informations")

config = parser.parse_args()

print("""
left click      draw tool
right click     erase tool or rescale an object
middle click    move an object
d or D          add inclusion or exclusion disk
c or C          add inclusion or exclusion circle
t or T          add inclusion or exclusion segment (click to validate end position)
Suppr           remove inclusion/exclusion object
p or P          add a particle (on given position or sticked to the interface)
+/-             increase or decrease iteration per frame (0 <=> pause)
i               display simulation and performance informations on the figure
r               start/stop recording
""")

# Device
import torch
if config.gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Loading model
from nnpf.problems import Problem
model = Problem.load_from_checkpoint(config.checkpoint, map_location=device)
model.freeze()
model.to(device)

# Domain
from nnpf.domain import Domain
if config.bounds is None:
    domain = model.domain
else:
    N = [int(round((b - a) / dx.item())) for (a, b), dx in zip(config.bounds, model.domain.dX)]
    bounds = [
        ((b - a) / 2 - n // 2 * dx.item(), (b - a) / 2 + (n - n // 2) * dx.item())
        for (a, b), dx, n in zip(config.bounds, model.domain.dX, N)
    ]
    domain = Domain(bounds, N, device=device)
print(f"domain = {domain}")

# Shapes and evolver
from utils import *
shapes = ShapeManager(domain, model.profil, model.hparams.epsilon)
evolver = Evolver(model, domain, shapes)
particles = ParticleManager(domain, model.iprofil, model.hparams.epsilon, oriented=shapes.oriented)
evolver.observers.append(particles.update)


# Output normalization
normalize = lambda u: (u - shapes.vout) / (shapes.vin - shapes.vout)

# Figure
import matplotlib.pyplot as plt
import nnpf.visu as visu
fig, ax = plt.subplots(figsize=(5, 5))
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("mycmap", ["red", "white", "blue", "green"])
img = ImShow(
    normalize(evolver.u).cpu(),
    X=[x.cpu() for x in domain.X],
    vmin=-1.,
    vmax=2.,
    cmap=cmap
)
p_graph, = plt.plot(*(x.cpu() for x in particles.X), color='black', linestyle="", marker='.', markersize=4)
infos_text = ax.text(0.5, 0.95, "", transform=ax.transAxes, ha="center")
plt.tight_layout()

# Events and performance metric
events = EventManager(ax.figure.canvas, config, shapes, particles, evolver)
perf = Performance()

def infos():
    text = f"t = {evolver.t:.1e} ; it = {evolver.iteration}"
    if perf.iteration > 0:
        text += f" ; {config.display_step}ipf ; {perf.fps:.1f}fps ; {perf.ips:.1f}ips"
    if events.recording:
        text += f" ; REC {events.recording_frame / events.recording_fps:.1f}s"
    return text

# Animation
def update(frame):
    events.on_new_frame()
    evolver.update(config.display_step)
    to_blit = img.update((normalize(evolver.u) + normalize(shapes.include) + normalize(shapes.exclude) - 1).cpu())
    p_graph.set_data(*(p.cpu() for p in particles.X))
    to_blit += p_graph,
    perf.new_frame(evolver.iteration)

    infos_str = infos()
    print(infos_str + " " * 20, end="\r", flush=True)
    if config.display_infos:
        infos_text.set_text(infos_str)
        to_blit += infos_text,
    elif infos_text.get_text() != "":
        infos_text.set_text("")
        to_blit += infos_text,

    return to_blit

from matplotlib.animation import FuncAnimation
anim = FuncAnimation(plt.gcf(), update, blit=True, interval=0)
plt.show()

print()

