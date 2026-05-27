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
        agent = Agent(
            self.env,
            params={
                "dt": self.dt,
                "speed_mean": config.SPEED_MEAN,
                "thigmotaxis": config.THIGMOTAXIS,
            }
        )

        n_steps = int(duration / self.dt)
        for _ in range(n_steps):
            agent.update()

        h = agent.get_history_arrays()

        pos_cm = h["pos"] * 100
        xs = pos_cm[:, 0]
        ys = pos_cm[:, 1]
        speeds = h["vel"]

        d_N = self.arena_cm - ys
        d_S = ys
        d_E = self.arena_cm - xs
        d_W = xs
        d_min = np.minimum(np.minimum(d_N, d_S), np.minimum(d_E, d_W))



        return {
            "x": xs, "y": ys,
            "speed": speeds,
            "d_N": d_N, "d_S": d_S, "d_E": d_E, "d_W": d_W,
            "d_min": d_min,
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
