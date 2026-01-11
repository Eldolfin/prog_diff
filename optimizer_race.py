import numpy as np
from manim import *
import math
import os
from surfaces import get_surface
from tensor import Tensor, sin_d, cos_d, exp_d


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
        surface_name = os.environ.get('SURFACE', 'funky')
        surf = get_surface(surface_name)
        
        # Set camera orientation based on surface
        self.set_camera_orientation(phi=surf.camera_phi * DEGREES, theta=surf.camera_theta * DEGREES, zoom=1.5)
        
        u_range = surf.u_range
        v_range = surf.v_range

        axes = ThreeDAxes(
            x_range=[u_range[0], u_range[1], 0.1],
            y_range=[v_range[0], v_range[1], 0.1],
            z_range=[surf.z_range[0], surf.z_range[1], 0.5],
            x_length=7, y_length=7, z_length=4
        ).shift(DOWN*1 + LEFT*0)

        surface = Surface(
            lambda u, v: axes.c2p(u, v, surf.func_numpy(u, v)),
            u_range=u_range,
            v_range=v_range,
            resolution=(64, 64),
            should_make_jagged=True 
        )
        
        surface.set_style(fill_opacity=0.8, stroke_color=BLUE_E, stroke_width=0.5)
        surface.set_fill_by_checkerboard(BLUE_D, BLUE_E, opacity=0.5)

        self.play(Create(axes), Create(surface))
        self.wait(1)

        start_u, start_v = surf.start_u, surf.start_v
        iterations = 120 

        competitors = [
            {"name": "SGD", "color": RED, "opt_class": SGD, "lr": 0.005},
            {"name": "Momentum (0.9)", "color": ORANGE, "opt_class": Momentum, "lr": 0.005, "kwargs": {"momentum": 0.9}},
            {"name": "RMSProp", "color": GREEN, "opt_class": RMSProp, "lr": 0.01, "kwargs": {"decay": 0.9}},
            {"name": "Adam", "color": YELLOW, "opt_class": Adam, "lr": 0.01, "kwargs": {"beta1": 0.9, "beta2": 0.999}}
        ]

        all_paths_group = VGroup()
        stats_data = []

        for comp in competitors:
            u_t = Tensor(start_u)
            v_t = Tensor(start_v)
            kwargs = comp.get("kwargs", {})
            optimizer = comp["opt_class"]([u_t, v_t], learning_rate=comp["lr"], **kwargs)
            path_points = []
            
            # Statistics tracking
            loss_values = []
            distances = []
            start_pos = np.array([start_u, start_v])
            
            for iter_num in range(iterations):
                curr_u = max(u_range[0], min(u_range[1], u_t.data))
                curr_v = max(v_range[0], min(v_range[1], v_t.data))
                curr_z = surf.func_numpy(curr_u, curr_v)
                path_points.append(axes.c2p(curr_u, curr_v, curr_z))
                
                # Track statistics
                loss_values.append(curr_z)
                curr_pos = np.array([curr_u, curr_v])
                distances.append(np.linalg.norm(curr_pos - start_pos))
                
                optimizer.zero_grad()
                loss = surf.func_tensor(u_t, v_t)
                loss.backward()
                optimizer.step()
            
            # Calculate final statistics
            final_loss = loss_values[-1]
            total_distance = sum([
                np.linalg.norm(np.array([path_points[i+1][0], path_points[i+1][1]]) - 
                              np.array([path_points[i][0], path_points[i][1]]))
                for i in range(len(path_points)-1)
            ])
            improvement = ((loss_values[0] - final_loss) / loss_values[0] * 100) if loss_values[0] != 0 else 0
            
            # Find convergence iteration (when loss stops improving significantly)
            convergence_iter = iterations
            threshold = 0.001
            for i in range(1, len(loss_values)):
                if abs(loss_values[i] - loss_values[i-1]) < threshold:
                    convergence_iter = i
                    break
            
            stats_data.append({
                "name": comp["name"],
                "color": comp["color"],
                "final_loss": final_loss,
                "total_distance": total_distance,
                "improvement": improvement,
                "convergence_iter": convergence_iter
            })
            
            comp["stats"] = stats_data[-1]

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
        
        # Create statistics display - more compact vertical layout
        stats_title = Text("Stats", font_size=24, weight=BOLD).to_corner(UL).shift(DOWN*0.15 + RIGHT*0.1)
        self.add_fixed_in_frame_mobjects(stats_title)
        
        stats_group = VGroup()
        for i, stat in enumerate(stats_data):
            # Name and all stats on separate lines for compactness
            name_text = Text(stat["name"], color=stat["color"], font_size=16, weight=BOLD)
            stats_text = Text(
                f"Loss: {stat['final_loss']:.3f} Dist: {stat['total_distance']:.1f}\nImprove: {stat['improvement']:.0f}% Conv: {stat['convergence_iter']}", 
                font_size=12,
                line_spacing=0.8
            )
            
            # Stack vertically
            opt_group = VGroup(name_text, stats_text)
            opt_group.arrange(DOWN, buff=0.05, aligned_edge=LEFT)
            stats_group.add(opt_group)
        
        stats_group.arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        stats_group.next_to(stats_title, DOWN, buff=0.15)
        stats_group.scale(0.85)
        self.add_fixed_in_frame_mobjects(stats_group)
        
        # Show all paths together with statistics
        self.play(
            FadeIn(stats_title),
            FadeIn(stats_group),
            Create(all_paths_group),
            run_time=2
        )
        
        self.move_camera(phi=75 * DEGREES, theta=-30 * DEGREES, run_time=3)
        self.wait(3)