"""RatInABox trajectory generation."""

import numpy as np
import config
from ratinabox.Environment import Environment
from ratinabox.Agent import Agent


class TrajectoryGenerator:
    """Generates trajectories in a 1×1 m arena using RatInABox."""

    def __init__(self, seed: int = None):
        self.seed = seed if seed is not None else config.RANDOM_SEED
        np.random.seed(self.seed)
        self.dt = config.TRAJECTORY_DT
        self.arena_cm = config.ARENA_CM

        self.env = Environment(
            params={
                "aspect": 1,
                "scale": config.ARENA_SIZE,
            }
        )

    def generate(self, duration: float) -> dict:
        """Generate a single trajectory.

        Returns dict with keys:
            x, y — positions (cm), shape (n_steps,)
            speed — speed magnitude (cm/s)
            head_direction — radians
            d_N, d_S, d_E, d_W — distances to each wall (cm)
            d_min — distance to nearest wall (cm)
            t — timestamps (s)
        """
        n_steps = int(duration / self.dt)

        agent = Agent(
            self.env,
            params={
                "dt": self.dt,
                "speed_mean": config.SPEED_MEAN,
                "thigmotaxis": config.THIGMOTAXIS,
            }
        )

        xs = np.zeros(n_steps)
        ys = np.zeros(n_steps)
        speeds = np.zeros(n_steps)
        head_dirs = np.zeros(n_steps)
        d_N = np.zeros(n_steps)
        d_S = np.zeros(n_steps)
        d_E = np.zeros(n_steps)
        d_W = np.zeros(n_steps)
        d_min = np.zeros(n_steps)

        for i in range(n_steps):
            agent.update()
            x = agent.pos[0] * 100
            y = agent.pos[1] * 100
            v = agent.velocity

            xs[i] = x
            ys[i] = y
            speeds[i] = np.sqrt(v[0]**2 + v[1]**2) * 100

            hd_val = agent.head_direction
            if isinstance(hd_val, (np.ndarray, list)):
                hd_val = float(hd_val[0] if len(hd_val) > 0 else 0.0)
            head_dirs[i] = float(hd_val)

            d_N[i] = self.arena_cm - y
            d_S[i] = y
            d_E[i] = self.arena_cm - x
            d_W[i] = x
            d_min[i] = min(d_N[i], d_S[i], d_E[i], d_W[i])

        t = np.arange(n_steps) * self.dt


        return {
            "x": xs, "y": ys,
            "speed": speeds,
            "head_direction": head_dirs,
            "d_N": d_N, "d_S": d_S, "d_E": d_E, "d_W": d_W,
            "d_min": d_min,
            "t": t,
        }


def generate_trajectory_batch(gen: TrajectoryGenerator,
                               duration: float,
                               n_trajectories: int = 1) -> dict:
    """Generate multiple trajectories, stacked along batch axis."""
    trajs = [gen.generate(duration) for _ in range(n_trajectories)]
    min_len = min(len(t["t"]) for t in trajs)
    for t in trajs:
        for k in t:
            t[k] = t[k][:min_len]
    result = {}
    for key in trajs[0]:
        result[key] = np.stack([t[key] for t in trajs], axis=0)
    return result
