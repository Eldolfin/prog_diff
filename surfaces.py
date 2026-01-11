import numpy as np
import math
from tensor import Tensor, sin_d, cos_d, exp_d


class Surface:
    
    def __init__(self, name, u_range, v_range, start_u, start_v, z_range=[-1, 1], camera_phi=60, camera_theta=-80):
        self.name = name
        self.u_range = u_range
        self.v_range = v_range
        self.start_u = start_u
        self.start_v = start_v
        self.z_range = z_range
        self.camera_phi = camera_phi  # In degrees
        self.camera_theta = camera_theta  # In degrees
    
    def func_numpy(self, u, v):
        raise NotImplementedError
    
    def func_tensor(self, u, v):
        raise NotImplementedError


class FunkySurface(Surface):    
    def __init__(self):
        super().__init__(
            name="Funky Surface",
            u_range=[0, 1.2],
            v_range=[0, 1.2],
            start_u=0.1,
            start_v=0.45
        )
    
    def func_numpy(self, u, v):
        val = 0.3 * (
            5 * u * v * (1 - u) * (1 - v) * np.cos(10 * v) * np.sin(10 * u * v) * np.exp(u)
            + np.exp(-((v - 0.4) ** 2 + (u - 0.2) ** 2) / 0.03)
            + 0.6 * np.exp(-((v - 0.4) ** 2) / 0.03) * np.sin(25*u) * (1-u)**2
            + 1.4 * np.exp(-(v - 0.6)**2 / 0.02) * (np.exp(-(u-0.7)**2 / 0.02) - np.exp(-(u-0.4)**2 / 0.02))
        )
        return val
    
    def func_tensor(self, u, v):
        term1 = Tensor(5) * u * v * (1 - u) * (1 - v) * cos_d(Tensor(10) * v) * sin_d(Tensor(10) * u * v) * exp_d(u)
        term2 = exp_d(-((v - 0.4) ** 2 + (u - 0.2) ** 2) / 0.03)
        term3 = Tensor(0.6) * exp_d(-((v - 0.4) ** 2) / 0.03) * sin_d(Tensor(25)*u) * (1-u)**2
        term4 = Tensor(1.4) * exp_d(-(v - 0.6)**2 / 0.02) * (exp_d(-(u-0.7)**2 / 0.02) - exp_d(-(u-0.4)**2 / 0.02))
        return Tensor(0.3) * (term1 + term2 + term3 + term4)


class Funky2Surface(Surface):    
    def __init__(self):
        super().__init__(
            name="Funky2 Surface",
            u_range=[0, 1.2],
            v_range=[0, 1.2],
            start_u=0.1,
            start_v=0.45
        )
    
    def func_numpy(self, u, v):
        return 1.0 * v * np.sin(2 * np.pi * (u - 0.6) / 1.2 - np.pi/2)
    
    def func_tensor(self, u, v):
        return Tensor(1.0) * v * sin_d(Tensor(2 * math.pi) * (u - Tensor(0.6)) / Tensor(1.2) - Tensor(np.pi/2))


class Funky3Surface(Surface):    
    def __init__(self):
        super().__init__(
            name="Funky3 Surface",
            u_range=[0, 1.2],
            v_range=[0, 1.2],
            start_u=0.2,
            start_v=0.1
        )
    
    def func_numpy(self, u, v):
        # Centerline of the path: a diagonal valley with a slight curve
        u0 = 0.2 + 0.6 * v  # path roughly from (0.2, 0) to (1.0, 1.0)
        
        # Squared distance to the path (small near the path, large away from it)
        dist2 = (u - u0)**2 + 0.05 * (v - 1.0)**2
        
        # Main valley: low along the path, higher as you move away
        valley = dist2
        
        # Gentle global bowl pulling everything toward the end of the path at (1, 1)
        bowl = 0.4 * ((u - 1.0)**2 + (v - 1.0)**2)
        
        # Small ripples along the path to make optimizer behavior visible but not chaotic
        ripples = 0.05 * np.sin(8 * u) * np.cos(6 * v)
        
        # Combine terms; add a constant so values are mostly positive
        return valley + bowl + ripples + 0.1
    
    def func_tensor(self, u, v):
        # Tensor version of the same surface
        u0 = Tensor(0.2) + Tensor(0.6) * v
        
        dist2 = (u - u0)**2 + Tensor(0.05) * (v - Tensor(1.0))**2
        valley = dist2
        
        bowl = Tensor(0.4) * ((u - Tensor(1.0))**2 + (v - Tensor(1.0))**2)
        
        ripples = Tensor(0.05) * sin_d(Tensor(8) * u) * cos_d(Tensor(6) * v)
        
        return valley + bowl + ripples + Tensor(0.1)


class SlopeSurface(Surface):    
    def __init__(self):
        super().__init__(
            name="Slope",
            u_range=[0, 2],
            v_range=[0, 2],
            start_u=1.8,
            start_v=1.8,
            z_range=[0, 2],
            camera_phi=70,
            camera_theta=-45
        )
    
    def func_numpy(self, u, v):
        return 0.5 * ((u - 1)**2 + (v - 1)**2)
    
    def func_tensor(self, u, v):
        return Tensor(0.5) * ((u - 1)**2 + (v - 1)**2)


class RosenbrockSurface(Surface):
    """Classic Rosenbrock banana function"""
    
    def __init__(self):
        super().__init__(
            name="Rosenbrock",
            u_range=[-2, 2],
            v_range=[-1, 3],
            start_u=-1.5,
            start_v=2.5,
            z_range=[0, 8],
            camera_phi=65,
            camera_theta=-70
        )
    
    def func_numpy(self, u, v):
        # Scale down for better visualization
        return 0.01 * ((1 - u)**2 + 100 * (v - u**2)**2)
    
    def func_tensor(self, u, v):
        return Tensor(0.01) * ((1 - u)**2 + Tensor(100) * (v - u**2)**2)


class RastriginSurface(Surface):
    """Rastrigin function - many local minima"""
    
    def __init__(self):
        super().__init__(
            name="Rastrigin",
            u_range=[-2.5, 2.5],
            v_range=[-2.5, 2.5],
            start_u=1.0,
            start_v=1.0,
            z_range=[0, 0.5],
            camera_phi=60,
            camera_theta=-80
        )
    
    def func_numpy(self, u, v):
        A = 10
        return 0.005 * (2*A + u**2 - A*np.cos(2*np.pi*u) + v**2 - A*np.cos(2*np.pi*v))
    
    def func_tensor(self, u, v):
        A = Tensor(10)
        return Tensor(0.005) * (Tensor(2)*A + u**2 - A*cos_d(Tensor(2*np.pi)*u) + v**2 - A*cos_d(Tensor(2*np.pi)*v))


class BealeSurface(Surface):
    """Beale function - challenging optimization surface"""
    
    def __init__(self):
        super().__init__(
            name="Beale",
            u_range=[-2, 2],
            v_range=[-2, 2],
            start_u=-1.5,
            start_v=1.5,
            z_range=[0, 6],
            camera_phi=65,
            camera_theta=-75
        )
    
    def func_numpy(self, u, v):
        # Scale down for visualization
        term1 = (1.5 - u + u*v)**2
        term2 = (2.25 - u + u*v**2)**2
        term3 = (2.625 - u + u*v**3)**2
        return 0.01 * (term1 + term2 + term3)
    
    def func_tensor(self, u, v):
        term1 = (Tensor(1.5) - u + u*v)**2
        term2 = (Tensor(2.25) - u + u*v**2)**2
        term3 = (Tensor(2.625) - u + u*v**3)**2
        return Tensor(0.01) * (term1 + term2 + term3)


class HimmelblauSurface(Surface):
    """Himmelblau function - four identical local minima"""
    
    def __init__(self):
        super().__init__(
            name="Himmelblau",
            u_range=[-4, 4],
            v_range=[-4, 4],
            start_u=3.5,
            start_v=3.5,
            z_range=[0, 8],
            camera_phi=65,
            camera_theta=-70
        )
    
    def func_numpy(self, u, v):
        return 0.01 * ((u**2 + v - 11)**2 + (u + v**2 - 7)**2)
    
    def func_tensor(self, u, v):
        return Tensor(0.01) * ((u**2 + v - Tensor(11))**2 + (u + v**2 - Tensor(7))**2)


class AckleySurface(Surface):
    """Ackley function - many local minima with one global minimum"""
    
    def __init__(self):
        super().__init__(
            name="Ackley",
            u_range=[-3, 3],
            v_range=[-3, 3],
            start_u=2.5,
            start_v=2.5,
            z_range=[-1, 2],
            camera_phi=60,
            camera_theta=-80
        )
    
    def func_numpy(self, u, v):
        term1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (u**2 + v**2)))
        term2 = -np.exp(0.5 * (np.cos(2*np.pi*u) + np.cos(2*np.pi*v)))
        return 0.1 * (term1 + term2 + 20 + np.e)
    
    def func_tensor(self, u, v):
        term1 = Tensor(-20) * exp_d(Tensor(-0.2) * (Tensor(0.5) * (u**2 + v**2))**Tensor(0.5))
        term2 = -exp_d(Tensor(0.5) * (cos_d(Tensor(2*np.pi)*u) + cos_d(Tensor(2*np.pi)*v)))
        return Tensor(0.1) * (term1 + term2 + Tensor(20 + np.e))


SURFACES = {
    'funky': FunkySurface(),
    'funky2': Funky2Surface(),
    'funky3': Funky3Surface(),
    'slope': SlopeSurface(),
    'rosenbrock': RosenbrockSurface(),
    'rastrigin': RastriginSurface(),
    'beale': BealeSurface(),
    'himmelblau': HimmelblauSurface(),
    'ackley': AckleySurface(),
}


def get_surface(name='funky'):
    if name not in SURFACES:
        raise ValueError(f"Unknown surface: {name}. Available: {list(SURFACES.keys())}")
    return SURFACES[name]
