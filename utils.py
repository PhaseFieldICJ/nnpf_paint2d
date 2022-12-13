from dataclasses import dataclass
import time
import torch
import math

class Evolver:
    def __init__(self, model, domain, shapes):
        self.model = model
        self.domain = domain
        self.u = torch.full(self.domain.N, shapes.vout, device=self.domain.device)
        self.shapes = shapes
        self.t = 0
        self.iteration = 0
        self.observers = []
        self.proj()

    def proj(self):
        self.shapes.inclusion_op(self.u, self.shapes.include, out=self.u)
        self.shapes.exclusion_op(self.u, self.shapes.exclude, out=self.u)

    def update(self, count=1):
        if count == 0:
            self.proj()
            return

        for i in range(count):
            self.u = self.model(self.u[None, None, ...]).squeeze()
            self.proj()
            self.t += self.model.hparams.dt
            self.iteration += 1
            for obs in self.observers:
                obs(self.u)


@dataclass
class Shape:
    func: 'typing.Any'
    center: tuple[float]
    radius: float
    include: bool

    def __call__(self, *X):
        return self.func(self.center, self.radius, *X)


class ShapeManager:
    def __init__(self, domain, profil, epsilon, margin=5):
        self.domain = domain
        self.profil = profil
        self.epsilon = epsilon
        self.margin = margin
        self.shapes = []

        # FIXME: ugly trick to detect if we are in oriented or non-oriented case
        extremal_values = self.profil(torch.tensor([float("-inf"), 0, float("inf")]), self.epsilon)
        if torch.isclose(extremal_values[0], extremal_values[2]):
            self.vin = extremal_values[1].item()
            self.din = 0
            self.oriented = False
        else:
            self.vin = extremal_values[0].item()
            self.din = float("-inf")
            self.oriented = True
        self.vout = extremal_values[2].item()

        self.update()

    def _init_mask(self):
        include = torch.full(self.domain.N, self.vout, device=self.domain.device)
        exclude = torch.full(self.domain.N, self.vin, device=self.domain.device)
        shape_id = torch.full(self.domain.N, -1, dtype=torch.int, device=self.domain.device)
        return include, exclude, shape_id

    def inclusion_op(self, a, b, out=None):
        if self.vin < self.vout:
            return torch.minimum(a, b, out=out)
        else:
            return torch.maximum(a, b, out=out)

    def exclusion_op(self, a, b, out=None):
        if self.vin < self.vout:
            return torch.maximum(a, b, out=out)
        else:
            return torch.minimum(a, b, out=out)

    def update(self):
        self.include, self.exclude, self.id = self._init_mask()
        from nnpf.shapes import periodic
        for i, shape in enumerate(self.shapes):
            dist = periodic(shape, self.domain.bounds)(*self.domain.X)
            if shape.include:
                self.inclusion_op(self.include, self.profil(dist, self.epsilon), out=self.include)
            else:
                self.exclusion_op(self.exclude, self.vin - self.profil(dist, self.epsilon), out=self.exclude)
            self.id[dist - min(self.domain.dX) * self.margin <= 0] = i

    def shape_id_at(self, *pos):
        return self.id[self.domain.index(*pos)]

    def add_shape(self, func, center, radius, include=True):
        self.shapes.append(Shape(func, center, radius, include))
        self.update()
        return len(self.shapes) - 1

    def del_shape(self, shape_id):
        if shape_id == -1:
            return
        del self.shapes[shape_id]
        self.update()

    def del_shape_at(self, *pos):
        self.del_shape(self.shape_id_at(*pos))

    def add_disk(self, center, radius=0, include=True):
        from nnpf.shapes import sphere
        def shape(center, radius, *X):
            return torch.clamp(sphere(radius, center)(*X), min=self.din)
        return self.add_shape(shape, center, radius, include)

    def add_circle(self, center, radius=0, include=True):
        from nnpf.shapes import sphere, unsign
        def shape(center, radius, *X):
            return unsign(sphere(radius, center))(*X)
        return self.add_shape(shape, center, radius, include)

    def add_segment(self, a, b, radius=0, include=True):
        from nnpf.shapes import segment, rounding
        def shape(center, radius, *X):
            return torch.clamp(
                rounding(
                    segment(
                        (a[0] + center[0], a[1] + center[1]),
                        (b[0] + center[0], b[1] + center[1])
                    ), abs(radius)
                )(*X), min=self.din)
        return self.add_shape(shape, [0, 0], radius, include)

    def __len__(self):
        return len(self.shapes)

    def __getitem__(self, idx):
        return self.shapes[idx]


def linear_interpolation(u, domain, X):
    right_index = domain.index(*X)
    left_index = [i - 1 for i in right_index] # Using negative index periodicity
    right_pos = [x[right_index] for x in domain.X] # Using right positions to avoid problem with periodicity when calculating alpha
    alpha = [(b - x) / dx for x, b, dx in zip(X, right_pos, domain.dX)]
    result = torch.zeros_like(X[0])
    for i in range(2):
        alpha_i = alpha[0] if i == 0 else 1 - alpha[0]
        for j in range(2):
            alpha_j = alpha[1] if j == 0 else 1 - alpha[1]
            result += alpha_i * alpha_j * u[(left_index[0] + i, left_index[1] + j)]

    return result


class ParticleManager:
    def __init__(self, domain, iprofil, epsilon, oriented=True, max_particles=1000):
        self.domain = domain
        self.iprofil = iprofil
        self.epsilon = epsilon
        self.oriented = oriented
        if oriented:
            self.dprofil = lambda u: 0.25 * (torch.tanh(u / 2)**2 - 1)
        else:
            self.dprofil = lambda u: 0.25 * torch.tanh(u / 2) * (1 - torch.tanh(u / 2)**2)

        self._X = [torch.zeros(max_particles, device=self.domain.device) for x in domain.X]
        self._distances = torch.zeros(max_particles, device=self.domain.device)
        self.cnt = 0
        self.last_u = None

    @property
    def X(self):
        return [x[:self.cnt] for x in self._X]

    @property
    def distances(self):
        return self._distances[:self.cnt]

    def add(self, pos, u, on_interface=False):
        assert self.cnt < self._distances.numel(), "To many particles!!!"

        for i, x in enumerate(pos):
            self._X[i][self.cnt] = x

        if on_interface:
            self._distances[self.cnt] = 0
        else:
            self._distances[self.cnt] = self.iprofil(self._clamp(u[self.domain.index(*pos)]), self.epsilon)

        self.cnt += 1

    def _clamp(self, u):
        """ To fix model output in order to get valid profil reciprocal values """
        if self.oriented:
            return u.clamp(0., 1.)
        else:
            return u.clamp(-0.25, 0.)

    def update(self, u):
        if self.last_u is None:
            self.last_u = u

        if self.cnt == 0:
            return

        index = self.domain.index(*self.X)

        gradient = [torch.roll(self.last_u, -1, d) - torch.roll(self.last_u, 1, d) for d in range(len(self.domain.X))]
        gradient = [g[index] for g in gradient]
        gradient_norm = torch.sqrt(sum(g**2 for g in gradient))
        mask = gradient_norm > 0
        directions = tuple(g[mask]/gradient_norm[mask] for g in gradient)

        #curr_values = u[index][mask] # FIXME: could be interpolated
        curr_values = linear_interpolation(u, self.domain, [x[mask] for x in self.X])
        lengths = self.iprofil(self._clamp(curr_values), self.epsilon) - self.distances[mask]
        lengths *= -self.dprofil(self.iprofil(self._clamp(curr_values), self.epsilon)).sign()

        # Avoid inf and NaN (extremal values or lack of precision)
        submask = lengths.isfinite()
        mask[mask == True] = submask


        for i, (x, d) in enumerate(zip(self.X, directions)):
            x[mask] += lengths[submask] * d[submask]
            a, b = self.domain.bounds[i]
            x[mask] = a + (x[mask] - a).remainder(b - a)

        self.last_u = u.clone()



class Performance:
    def __init__(self, length=30):
        self.times = [time.perf_counter()] * length
        self.iterations = [0] * length
        self.idx = 0

    def _next_idx(self):
        return(self.idx + 1) % len(self.times)

    def new_frame(self, iteration):
        self.idx = self._next_idx()
        self.times[self.idx] = time.perf_counter()
        self.iterations[self.idx] = iteration

    @property
    def iteration(self):
        return self.iterations[self.idx]

    @property
    def fps(self):
        dt = self.times[self.idx] - self.times[self._next_idx()]
        return len(self.times) / dt

    @property
    def ips(self):
        dt = self.times[self.idx] - self.times[self._next_idx()]
        return (self.iterations[self.idx] - self.iterations[self._next_idx()]) / dt


# Events manager
class EventManager:
    def __init__(self, canvas, config, shapes, particles, evolver):
        self.config = config
        self.shapes = shapes
        self.particles = particles
        self.evolver = evolver
        self.action = None
        self.recording = False
        canvas.mpl_connect('key_press_event', self.on_key_press)
        canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        canvas.mpl_connect('button_press_event', self.on_button_press)
        canvas.mpl_connect('button_release_event', self.on_button_release)

    def on_key_press(self, event):
        #print(f"press {event.key}")
        if not event.inaxes:
            return
        self.key_press_pos = event.xdata, event.ydata

        if event.key.lower() == 'd':
            self.shapes.add_disk(self.key_press_pos, self.evolver.model.hparams.epsilon, event.key.islower())
        if event.key.lower() == 'c':
            self.shapes.add_circle(self.key_press_pos, self.evolver.model.hparams.epsilon, event.key.islower())
        elif event.key.lower() == 't':
            self.action = "segment"
            self.segment_include = event.key.islower()
        elif event.key == "delete":
            self.shapes.del_shape_at(*self.key_press_pos)
        elif event.key == "+":
            self.config.display_step += 1
        elif event.key == "-":
            self.config.display_step = max(0, self.config.display_step - 1)
        elif event.key == "r":
            if self.recording:
                self.anim_writer.close()
            else:
                from nnpf.visu import AnimWriter
                from datetime import datetime
                self.recording_frame = 0
                self.recording_fps = 25
                file_name = f"record_{datetime.now().isoformat(timespec='seconds')}.avi"
                output_params = ['-preset', 'fast', '-tune', 'animation', '-crf', '25', '-threads', '6']
                self.anim_writer = AnimWriter(file_name, fps=self.recording_fps, output_params=output_params)
            self.recording = not self.recording
        elif event.key == "i":
            self.config.display_infos = not self.config.display_infos
        elif event.key.lower() == "p":
            self.particles.add(self.key_press_pos, self.evolver.u, not event.key.islower())


    def on_button_press(self, event):
        if not event.inaxes or self.action is not None:
            return
        self.button_press_pos = event.xdata, event.ydata

        if event.button == 1:
            self.action = "draw"
            self.pen_shape_id = self.shapes.add_disk(self.button_press_pos, self.evolver.model.hparams.epsilon, True)

        elif event.button == 2:
            shape_id = self.shapes.shape_id_at(*self.button_press_pos)
            if shape_id >= 0:
                self.action = "move"
                self.shape_id = shape_id
                self.shape_pos = self.shapes[shape_id].center

        elif event.button == 3:
            shape_id = self.shapes.shape_id_at(*self.button_press_pos)
            if shape_id == -1:
                self.action = "erase"
                self.pen_shape_id = self.shapes.add_disk(self.button_press_pos, self.evolver.model.hparams.epsilon, False)
            else:
                self.action = "radius"
                self.shape_id = shape_id
                self.shape_radius = self.shapes[shape_id].radius

    def on_button_release(self, event):
        self.button_release_pos = event.xdata, event.ydata
        if self.action is None:
            return

        if event.button == 1:
            if self.action in ["draw"]:
                self.action = None
                self.shapes.del_shape(self.pen_shape_id)
            elif self.action == "segment":
                self.shapes.add_segment(self.key_press_pos, self.button_release_pos, self.evolver.model.hparams.epsilon, self.segment_include)
                self.action = None

        elif event.button == 2:
            if self.action in ["move"]:
                self.action = None

        elif event.button == 3:
            if self.action in ["erase", "radius"]:
                if self.action == "erase":
                    self.shapes.del_shape(self.pen_shape_id)
                self.action = None

    def on_mouse_move(self, event):
        if not event.inaxes:
            return
        self.move_pos = event.xdata, event.ydata

        if self.action in ["draw", "erase"]:
            self.shapes[self.pen_shape_id].center = self.move_pos
            self.shapes.update()

        elif self.action == "move":
            self.shapes[self.shape_id].center = [self.shape_pos[i] + self.move_pos[i] - self.button_press_pos[i] for i in range(2)]
            self.shapes.update()

        elif self.action == "radius":
            press_dist = math.sqrt(sum((a - b)**2 for a, b in zip(self.button_press_pos, self.shapes[self.shape_id].center)))
            move_dist = math.sqrt(sum((a - b)**2 for a, b in zip(self.move_pos, self.shapes[self.shape_id].center)))
            self.shapes[self.shape_id].radius = self.shape_radius + move_dist - press_dist
            self.shapes.update()

    def on_new_frame(self):
        if self.recording:
            self.anim_writer.add_frame()
            self.recording_frame += 1


from nnpf.visu import ImShow
class ImShowCache(ImShow):
    def __init__(self, img, *args, **kwargs):
        super().__init__(img.cpu(), *args, **kwargs)
        self.img = img.clone()

    def update(self, img):
        if torch.equal(img, self.img):
            return ()
        self.img = img.clone()
        return super().update(img.cpu())

