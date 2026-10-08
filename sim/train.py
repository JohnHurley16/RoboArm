"""Train one agent, or a whole population of different algorithms at once.

One run:
    python sim/train.py --task throw --algo brax_ppo --steps 200e6 --tag ppo_a

A population (every run in parallel, sharing the GPU; see sim/populations/):
    python sim/train.py --population sim/populations/throw.json
    python sim/leaderboard.py runs/throw

Algorithms
    brax_ppo, brax_sac      Brax trainers: the whole loop on the GPU, thousands of envs
    sb3_ppo, sb3_sac, sb3_td3, sb3_tqc*
                            Stable-Baselines3 on the same MJX envs via sim/sb3_vec.py
                            (*tqc needs sb3-contrib)

Every run writes runs/<group>/<tag>/: config.json, metrics.csv (one row per
evaluation), and the policy checkpoint (policy.pkl for Brax, model.zip for SB3).
"""
import argparse
import csv
import json
import os
import pickle
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import jax_compat  # noqa: E402,F401  (Brax on new JAX)
ROOT = os.path.dirname(HERE)

# sensible starting points per task (Brax PPO)
PPO_DEFAULTS = dict(
    reach=dict(num_envs=2048, num_timesteps=30e6, unroll_length=20, batch_size=512, num_minibatches=32,
               num_updates_per_batch=4, discounting=0.97, learning_rate=3e-4, entropy_cost=1e-2),
    lift=dict(num_envs=4096, num_timesteps=150e6, unroll_length=20, batch_size=512, num_minibatches=32,
              num_updates_per_batch=4, discounting=0.97, learning_rate=3e-4, entropy_cost=1e-2),
    place=dict(num_envs=4096, num_timesteps=250e6, unroll_length=20, batch_size=512, num_minibatches=32,
               num_updates_per_batch=4, discounting=0.98, learning_rate=3e-4, entropy_cost=1e-2),
    throw=dict(num_envs=4096, num_timesteps=200e6, unroll_length=25, batch_size=512, num_minibatches=32,
               num_updates_per_batch=4, discounting=0.99, learning_rate=3e-4, entropy_cost=5e-3),
    pick_throw=dict(num_envs=4096, num_timesteps=500e6, unroll_length=25, batch_size=512, num_minibatches=32,
                    num_updates_per_batch=4, discounting=0.995, learning_rate=3e-4, entropy_cost=5e-3),
)
SAC_DEFAULTS = dict(num_envs=256, num_timesteps=20e6, batch_size=512, learning_rate=3e-4, discounting=0.98,
                    min_replay_size=8192, max_replay_size=2_000_000, grad_updates_per_step=32, reward_scaling=1.0)


def build_env(task, env_kw):
    """env_kw comes from JSON: {"bag": {...}} / {"board": {...}} become Bag / Board."""
    from envs import make_env
    from scene import Bag, Board
    kw = dict(env_kw)
    if isinstance(kw.get("bag"), dict):
        kw["bag"] = Bag(**kw["bag"])
    if isinstance(kw.get("board"), dict):
        kw["board"] = Board(**kw["board"])
    for k in ("distance_range", "yaw_range"):
        if k in kw:
            kw[k] = tuple(kw[k])
    return make_env(task, **kw)


class Logger:
    def __init__(self, out):
        self.path = os.path.join(out, "metrics.csv")
        self.t0 = time.time()
        self.rows = 0

    def __call__(self, step, metrics):
        row = {"step": int(step), "wall_s": round(time.time() - self.t0, 1)}
        row.update({k: float(v) for k, v in metrics.items() if hasattr(v, "__float__")})
        new = not os.path.exists(self.path)
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        keys = [k for k in row if k.endswith(("episode_reward", "points", "success", "dist", "in_hole"))][:4]
        print(f"[{os.path.basename(os.path.dirname(self.path))}] step {step:>12,.0f}  "
              + "  ".join(f"{k.split('/')[-1]} {row[k]:.3f}" for k in keys), flush=True)


def run_brax(cfg, out):
    import functools
    from brax.training.agents.ppo import networks as ppo_net
    from brax.training.agents.ppo import train as ppo
    from brax.training.agents.sac import train as sac
    env = build_env(cfg["task"], cfg.get("env", {}))
    eval_env = build_env(cfg["task"], dict(cfg.get("env", {}), dr=__import__("envs").DR(enabled=True)))
    log = Logger(out)
    hp = dict(cfg.get("hp", {}))
    if cfg["algo"] == "brax_ppo":
        base = dict(PPO_DEFAULTS[cfg["task"]])
        base.update(hp)
        layers = tuple(base.pop("layers", (256, 256, 256)))
        net = functools.partial(ppo_net.make_ppo_networks, policy_hidden_layer_sizes=layers,
                                value_hidden_layer_sizes=layers)
        ep_len = int(base.pop("episode_length", env.episode_length))
        make_inf, params, _ = ppo.train(
            environment=env, eval_env=eval_env, episode_length=ep_len, action_repeat=1,
            normalize_observations=True, reward_scaling=1.0, num_evals=int(base.pop("num_evals", 40)),
            network_factory=net, seed=cfg.get("seed", 0), progress_fn=log, **base)
    else:
        base = dict(SAC_DEFAULTS)
        base.update(hp)
        ep_len = int(base.pop("episode_length", env.episode_length))
        make_inf, params, _ = sac.train(
            environment=env, eval_env=eval_env, episode_length=ep_len, action_repeat=1,
            normalize_observations=True, num_evals=int(base.pop("num_evals", 40)),
            seed=cfg.get("seed", 0), progress_fn=log, **base)
    with open(os.path.join(out, "policy.pkl"), "wb") as f:
        pickle.dump(dict(params=params, algo=cfg["algo"], task=cfg["task"], env=cfg.get("env", {}),
                         layers=PPO_DEFAULTS.get(cfg["task"], {}).get("layers", (256, 256, 256))), f)


def run_sb3(cfg, out):
    import stable_baselines3 as sb3
    from stable_baselines3.common.callbacks import BaseCallback
    from sb3_vec import JaxVecEnv
    env = build_env(cfg["task"], cfg.get("env", {}))
    hp = dict(cfg.get("hp", {}))
    n_envs = int(hp.pop("num_envs", 256 if cfg["algo"] == "sb3_ppo" else 32))
    ep_len = hp.pop("episode_length", None)
    steps = int(float(hp.pop("num_timesteps", 20e6)))
    venv = JaxVecEnv(env, num_envs=n_envs, seed=cfg.get("seed", 0), episode_length=ep_len and int(ep_len))
    algo = cfg["algo"][4:]
    if algo == "tqc":
        from sb3_contrib import TQC as Algo
    else:
        Algo = getattr(sb3, algo.upper())
    defaults = dict(ppo=dict(n_steps=64, batch_size=4096, n_epochs=5, gamma=0.98, learning_rate=3e-4, ent_coef=0.005),
                    sac=dict(buffer_size=1_000_000, batch_size=512, gamma=0.98, train_freq=1, gradient_steps=4,
                             learning_starts=10_000),
                    td3=dict(buffer_size=1_000_000, batch_size=512, gamma=0.98, train_freq=1, gradient_steps=4,
                             learning_starts=10_000),
                    tqc=dict(buffer_size=1_000_000, batch_size=512, gamma=0.98, learning_starts=10_000))[algo]
    defaults.update(hp)
    log = Logger(out)

    class CB(BaseCallback):
        def __init__(self):
            super().__init__()
            self.buf, self.next = [], 0

        def _on_step(self):
            for info in self.locals["infos"]:
                if "episode" in info:
                    self.buf.append(info)
            if self.num_timesteps >= self.next and self.buf:
                keys = {k for b in self.buf for k in b if k.startswith("m_")}
                m = {"eval/episode_reward": sum(b["episode"]["r"] for b in self.buf) / len(self.buf)}
                m.update({f"eval/episode_{k[2:]}": sum(b.get(k, 0) for b in self.buf) / len(self.buf) for k in keys})
                log(self.num_timesteps, m)
                self.buf, self.next = [], self.num_timesteps + steps // 40
            return True

    model = Algo("MlpPolicy", venv, verbose=0, seed=cfg.get("seed", 0),
                 policy_kwargs=dict(net_arch=[256, 256, 256]), **defaults)
    model.learn(steps, callback=CB())
    model.save(os.path.join(out, "model.zip"))


def run_one(cfg):
    out = os.path.join(ROOT, "runs", cfg.get("group", cfg["task"]), cfg["tag"])
    os.makedirs(out, exist_ok=True)
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
    (run_brax if cfg["algo"].startswith("brax") else run_sb3)(cfg, out)
    print(f"[{cfg['tag']}] done -> {out}")


def run_population(path, max_parallel=None):
    pop = json.load(open(path))
    runs = pop["runs"]
    n = max_parallel or pop.get("max_parallel", len(runs))
    frac = pop.get("gpu_mem_fraction", round(0.9 / min(n, len(runs)), 2))
    procs, queue = [], list(runs)
    group = pop.get("group", os.path.splitext(os.path.basename(path))[0])
    os.makedirs(os.path.join(ROOT, "runs", group), exist_ok=True)
    while queue or procs:
        while queue and len(procs) < n:
            r = dict(pop.get("defaults", {}), **queue.pop(0))
            r["group"] = group
            env = dict(os.environ, XLA_PYTHON_CLIENT_MEM_FRACTION=str(frac), XLA_PYTHON_CLIENT_PREALLOCATE="false")
            log = open(os.path.join(ROOT, "runs", group, f"{r['tag']}.log"), "w")
            p = subprocess.Popen([sys.executable, __file__, "--config", json.dumps(r)], env=env, stdout=log,
                                 stderr=subprocess.STDOUT)
            procs.append((p, r["tag"]))
            print(f"started {r['tag']} ({r['algo']}, pid {p.pid})")
        time.sleep(10)
        for p, tag in list(procs):
            if p.poll() is not None:
                print(f"finished {tag} (exit {p.returncode})")
                procs.remove((p, tag))
    print(f"population done; python sim/leaderboard.py runs/{group}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["reach", "lift", "place", "throw", "pick_throw"])
    ap.add_argument("--algo", default="brax_ppo", choices=["brax_ppo", "brax_sac", "sb3_ppo", "sb3_sac", "sb3_td3", "sb3_tqc"])
    ap.add_argument("--steps", type=float, help="total env steps")
    ap.add_argument("--envs", type=int, help="parallel envs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag")
    ap.add_argument("--population", help="JSON file with many runs to train in parallel")
    ap.add_argument("--parallel", type=int, help="max runs at once (population mode)")
    ap.add_argument("--config", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.population:
        return run_population(a.population, a.parallel)
    if a.config:
        return run_one(json.loads(a.config))
    hp = {}
    if a.steps:
        hp["num_timesteps"] = a.steps
    if a.envs:
        hp["num_envs"] = a.envs
    run_one(dict(task=a.task, algo=a.algo, seed=a.seed, tag=a.tag or f"{a.algo}_s{a.seed}", hp=hp))


if __name__ == "__main__":
    main()
