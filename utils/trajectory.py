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
        vel_cms = h["vel"] * 100
        vx = vel_cms[:, 0]
        vy = vel_cms[:, 1]

        d_N = self.arena_cm - ys
        d_S = ys
        d_E = self.arena_cm - xs
        d_W = xs
        d_min = np.minimum(np.minimum(d_N, d_S), np.minimum(d_E, d_W))

        t = np.arange(n_steps) * self.dt
        head_direction = np.arctan2(vy, vx)

        return {
            "x": xs, "y": ys,
            "speed": np.sqrt(vx**2 + vy**2),
            "vx": vx, "vy": vy,
            "head_direction": head_direction,
            "d_N": d_N, "d_S": d_S, "d_E": d_E, "d_W": d_W,
            "d_min": d_min, "t": t,
        }


def interpolate_trajectory(traj: dict, target_dt: float) -> dict:
    """Linearly interpolate trajectory to a finer time resolution.

    x, y are linearly interpolated. vx, vy are step-constant (velocity of
    the coarse segment's left edge). d_* are recomputed from interpolated x, y.

    Args:
        traj: dict at config.TRAJECTORY_DT resolution (keys: x, y, vx, vy,
              d_N, d_S, d_E, d_W, d_min, speed, head_direction, t)
        target_dt: desired time step in seconds

    Returns:
        dict with same keys at target_dt resolution
    """
    old_t = traj["t"]
    total_time = old_t[-1]
    n_new = int(round(total_time / target_dt)) + 1
    new_t = np.arange(n_new, dtype=np.float64) * target_dt

    out = {}
    out["x"] = np.interp(new_t, old_t, traj["x"])
    out["y"] = np.interp(new_t, old_t, traj["y"])

    idx = np.searchsorted(old_t, new_t, side="right") - 1
    idx = np.clip(idx, 0, len(old_t) - 1)
    out["vx"] = traj["vx"][idx]
    out["vy"] = traj["vy"][idx]

    arena = config.ARENA_CM
    out["d_N"] = arena - out["y"]
    out["d_S"] = out["y"]
    out["d_E"] = arena - out["x"]
    out["d_W"] = out["x"]

    out["t"] = new_t
    out["d_min"] = np.minimum(np.minimum(out["d_N"], out["d_S"]),
                              np.minimum(out["d_E"], out["d_W"]))
    out["head_direction"] = np.arctan2(out["vy"], out["vx"])
    out["speed"] = np.sqrt(out["vx"]**2 + out["vy"]**2)

    return out


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
