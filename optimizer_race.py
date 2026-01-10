import numpy as np
from manim import *
import math

class Tensor:
    def __init__(self, data, _children=(), _op=''):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Tensor(self.data**other, (self,), f'**{other}')
        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward
        return out

    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Tensor(data={self.data:.4f}, grad={self.grad:.4f})"

def sin_d(x):
    out = Tensor(np.sin(x.data), (x,), 'sin')
    def _backward():
        x.grad += np.cos(x.data) * out.grad
    out._backward = _backward
    return out

def cos_d(x):
    out = Tensor(np.cos(x.data), (x,), 'cos')
    def _backward():
        x.grad += -np.sin(x.data) * out.grad
    out._backward = _backward
    return out

def exp_d(x):
    out = Tensor(np.exp(x.data), (x,), 'exp')
    def _backward():
        x.grad += out.data * out.grad
    out._backward = _backward
    return out

class Optimizer:
    def __init__(self, params):
        self.params = list(params)
    def zero_grad(self):
        for p in self.params:
            p.grad = 0.0

class SGD(Optimizer):
    def __init__(self, params, learning_rate=0.01):
        super().__init__(params)
        self.lr = learning_rate
    def step(self):
        for p in self.params:
            p.data -= self.lr * p.grad

class Momentum(Optimizer):
    def __init__(self, params, learning_rate=0.01, momentum=0.9):
        super().__init__(params)
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = [0.0 for _ in params]
    def step(self):
        for i, p in enumerate(self.params):
            self.velocity[i] = self.momentum * self.velocity[i] - self.lr * p.grad
            p.data += self.velocity[i]

class RMSProp(Optimizer):
    def __init__(self, params, learning_rate=0.01, decay=0.9, eps=1e-8):
        super().__init__(params)
        self.lr = learning_rate
        self.decay = decay
        self.eps = eps
        self.cache = [0.0 for _ in params]
    def step(self):
        for i, p in enumerate(self.params):
            self.cache[i] = self.decay * self.cache[i] + (1 - self.decay) * p.grad**2
            p.data -= (self.lr * p.grad) / (np.sqrt(self.cache[i]) + self.eps)

class Adam(Optimizer):
    def __init__(self, params, learning_rate=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(params)
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [0.0 for _ in params]
        self.v = [0.0 for _ in params]
        self.t = 0
    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * p.grad**2
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

class OptimizerRace(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=-80 * DEGREES, zoom=1.5)
        
        u_range = [0, 1.2]
        v_range = [0, 1.2]

        def funky_func_numpy(u, v):
            val = 0.3 * (
                5 * u * v * (1 - u) * (1 - v) * np.cos(10 * v) * np.sin(10 * u * v) * np.exp(u)
                + np.exp(-((v - 0.4) ** 2 + (u - 0.2) ** 2) / 0.03)
                + 0.6 * np.exp(-((v - 0.4) ** 2) / 0.03) * np.sin(25*u) * (1-u)**2
                + 1.4 * np.exp(-(v - 0.6)**2 / 0.02) * (np.exp(-(u-0.7)**2 / 0.02) - np.exp(-(u-0.4)**2 / 0.02))
            )
            return val

        def funky_func_tensor(u, v):
            term1 = Tensor(5) * u * v * (1 - u) * (1 - v) * cos_d(Tensor(10) * v) * sin_d(Tensor(10) * u * v) * exp_d(u)
            term2 = exp_d(-((v - 0.4) ** 2 + (u - 0.2) ** 2) / 0.03)
            term3 = Tensor(0.6) * exp_d(-((v - 0.4) ** 2) / 0.03) * sin_d(Tensor(25)*u) * (1-u)**2
            term4 = Tensor(1.4) * exp_d(-(v - 0.6)**2 / 0.02) * (exp_d(-(u-0.7)**2 / 0.02) - exp_d(-(u-0.4)**2 / 0.02))
            return Tensor(0.3) * (term1 + term2 + term3 + term4)

        axes = ThreeDAxes(
            x_range=[u_range[0], u_range[1], 0.1],
            y_range=[v_range[0], v_range[1], 0.1],
            z_range=[-1, 1, 0.5],
            x_length=7, y_length=7, z_length=4
        ).shift(DOWN*1 + LEFT*0)

        surface = Surface(
            lambda u, v: axes.c2p(u, v, funky_func_numpy(u, v)),
            u_range=u_range,
            v_range=v_range,
            resolution=(64, 64),
            should_make_jagged=True 
        )
        
        surface.set_style(fill_opacity=0.8, stroke_color=BLUE_E, stroke_width=0.5)
        surface.set_fill_by_checkerboard(BLUE_D, BLUE_E, opacity=0.5)

        self.play(Create(axes), Create(surface))
        self.wait(1)

        start_u, start_v = 0.1, 0.45 
        iterations = 120 

        competitors = [
            {"name": "SGD", "color": RED, "opt_class": SGD, "lr": 0.005},
            {"name": "Momentum (0.9)", "color": ORANGE, "opt_class": Momentum, "lr": 0.005, "kwargs": {"momentum": 0.9}},
            {"name": "RMSProp", "color": GREEN, "opt_class": RMSProp, "lr": 0.01, "kwargs": {"decay": 0.9}},
            {"name": "Adam", "color": YELLOW, "opt_class": Adam, "lr": 0.01, "kwargs": {"beta1": 0.9, "beta2": 0.999}}
        ]

        all_paths_group = VGroup()

        for comp in competitors:
            u_t = Tensor(start_u)
            v_t = Tensor(start_v)
            kwargs = comp.get("kwargs", {})
            optimizer = comp["opt_class"]([u_t, v_t], learning_rate=comp["lr"], **kwargs)
            path_points = []
            
            for _ in range(iterations):
                curr_u = max(u_range[0], min(u_range[1], u_t.data))
                curr_v = max(v_range[0], min(v_range[1], v_t.data))
                curr_z = funky_func_numpy(curr_u, curr_v)
                path_points.append(axes.c2p(curr_u, curr_v, curr_z))
                
                optimizer.zero_grad()
                loss = funky_func_tensor(u_t, v_t)
                loss.backward()
                optimizer.step()

            title = Text(comp["name"], font_size=48, color=comp["color"])
            title.to_corner(UL)
            self.add_fixed_in_frame_mobjects(title)
            self.play(Write(title), run_time=0.5)

            ball = Sphere(radius=0.1, color=comp["color"])
            ball.move_to(path_points[0])
            ball.set_z_index(10) 
            self.play(FadeIn(ball), run_time=0.5)

            path_mobj = VMobject()
            path_mobj.set_points_as_corners(path_points)
            path_mobj.set_color(comp["color"])
            path_mobj.set_stroke(width=4)
            path_mobj.set_z_index(5) 
            
            all_paths_group.add(path_mobj.copy())

            self.play(
                MoveAlongPath(ball, path_mobj),
                Create(path_mobj),
                run_time=4,
                rate_func=linear
            )
            
            self.wait(0.5)
            self.play(FadeOut(ball), FadeOut(path_mobj), FadeOut(title), run_time=0.5)

        self.wait(1)
        
        legend = VGroup()
        legend_title = Text("Final diff", font_size=36).to_corner(UL).shift(DOWN*0.5)
        self.add_fixed_in_frame_mobjects(legend_title)
        
        for i, comp in enumerate(competitors):
            l = Text(comp["name"], color=comp["color"], font_size=24)
            legend.add(l)
        
        legend.arrange(DOWN, aligned_edge=LEFT)
        legend.next_to(legend_title, DOWN)
        self.add_fixed_in_frame_mobjects(legend)
        
        self.play(
            FadeIn(legend_title),
            FadeIn(legend),
            Create(all_paths_group),
            run_time=2
        )
        
        self.move_camera(phi=75 * DEGREES, theta=-30 * DEGREES, run_time=3)
        self.wait(3)