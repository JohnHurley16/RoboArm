"""Evaluate a trained policy and render it.

    python sim/play.py runs/throw/ppo_base               # 1000 episodes of stats + a video of 4
    python sim/play.py runs/throw/ppo_base --episodes 200 --video 8 --distance 2.74

Works with Brax checkpoints (policy.pkl) and SB3 ones (model.zip).  Videos use
the full visual model (same joints and bodies as the training model).
"""
import argparse
import json
import math
import os
import pickle
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import jax_compat  # noqa: E402,F401  (Brax on new JAX)

import jax                                              # noqa: E402
import jax.numpy as jnp                                 # noqa: E402
import mujoco                                           # noqa: E402
import numpy as np                                      # noqa: E402

from envs import make_env, DR                           # noqa: E402
from scene import scene_xml                             # noqa: E402


def load_policy(run_dir, env):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    if os.path.exists(os.path.join(run_dir, "policy.pkl")):
        ck = pickle.load(open(os.path.join(run_dir, "policy.pkl"), "rb"))
        from brax.training.acme import running_statistics
        if ck["algo"] == "brax_ppo":
            from brax.training.agents.ppo import networks as N
            layers = tuple(cfg.get("hp", {}).get("layers", ck.get("layers", (256, 256, 256))))
            nets = N.make_ppo_networks(env.observation_size, env.action_size,
                                       preprocess_observations_fn=running_statistics.normalize,
                                       policy_hidden_layer_sizes=layers, value_hidden_layer_sizes=layers)
            make = N.make_inference_fn(nets)
        else:
            from brax.training.agents.sac import networks as N
            nets = N.make_sac_networks(env.observation_size, env.action_size,
                                       preprocess_observations_fn=running_statistics.normalize)
            make = N.make_inference_fn(nets)
        params = ck["params"]
        try:
            pi = make(params, deterministic=True)
        except Exception:                                      # older/newer Brax: (normalizer, policy) only
            pi = make(params[:2], deterministic=True)
        return jax.jit(jax.vmap(lambda o, k: pi(o, k)[0])), cfg
    # SB3
    import stable_baselines3 as sb3
    algo = cfg["algo"][4:]
    Algo = __import__("sb3_contrib").TQC if algo == "tqc" else getattr(sb3, algo.upper())
    model = Algo.load(os.path.join(run_dir, "model.zip"), device="cpu")
    return (lambda o, k: jnp.asarray(model.predict(np.asarray(o), deterministic=True)[0])), cfg


EVENTS = ("points", "in_hole", "on_board", "release_err")


def evaluate(env, policy, episodes=1000, seed=0, batch=256):
    reset, step = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))
    totals, n = {}, 0
    key = jax.random.PRNGKey(seed)
    while n < episodes:
        b = min(batch, episodes - n)
        key, k1 = jax.random.split(key)
        s = reset(jax.random.split(k1, b))
        done = jnp.zeros(b, bool)
        acc = {k: jnp.zeros(b) for k in s.metrics}
        ret, nsteps = jnp.zeros(b), jnp.zeros(b)
        for t in range(env.episode_length):
            key, k2 = jax.random.split(key)
            a = policy(s.obs, jax.random.split(k2, b))
            s = step(s, a)
            live = ~done
            ret = ret + s.reward * live
            nsteps = nsteps + live
            for k in acc:
                acc[k] = acc[k] + s.metrics[k] * live
            done = done | (s.done > 0.5)
            if bool(done.all()):
                break
        totals.setdefault("return", []).extend(np.asarray(ret).tolist())
        for k, v in acc.items():          # events (points, ...) are summed; states are averaged per step
            v = v if k in EVENTS else v / jnp.maximum(nsteps, 1)
            totals.setdefault(k, []).extend(np.asarray(v).tolist())
        n += b
    return {k: float(np.mean(v)) for k, v in totals.items()}


def render(env, policy, path, episodes=4, seed=1, size=(720, 1280)):
    full = mujoco.MjModel.from_xml_string(scene_xml(bag=env.bag_cfg, board=env.board_cfg,
                                                    target_site=hasattr(env, "task_obs") and env.__class__.__name__ == "Reach",
                                                    lean=False))
    data = mujoco.MjData(full)
    r = mujoco.Renderer(full, size[0], size[1])
    cam = mujoco.MjvCamera()
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    frames = []
    every = max(1, int(round((1 / 30) / env.ctrl_dt)))
    for ep in range(episodes):
        s = reset(jax.random.PRNGKey(seed + ep))
        info = s.info
        if "board_dist" in info:                    # move the visual board to this episode's spot
            mx = env.model_for(info)
            bid = full.body("board").id
            full.body_pos[bid] = np.asarray(mx.body_pos[env.board_body])
            full.body_quat[bid] = np.asarray(mx.body_quat[env.board_body])
            sk = full.geom("board_skirt").id
            full.geom_pos[sk] = np.asarray(mx.geom_pos[env.skirt_geom])
            full.geom_quat[sk] = np.asarray(mx.geom_quat[env.skirt_geom])
            cam.lookat[:] = [float(info["board_dist"]) * 0.5, 0, 0.25]
            cam.distance, cam.azimuth, cam.elevation = 3.2, -120, -15
        else:
            cam.lookat[:] = [0.2, 0, 0.2]
            cam.distance, cam.azimuth, cam.elevation = 1.3, -135, -20
        if "goal" in info and full.nmocap:
            data.mocap_pos[0] = np.asarray(info["goal"])
        for t in range(env.episode_length):
            a = policy(s.obs[None], jax.random.split(jax.random.PRNGKey(t), 1))[0]
            s = step(s, a)
            if t % every == 0:
                data.qpos[:] = np.asarray(s.pipeline_state.qpos)
                mujoco.mj_forward(full, data)
                r.update_scene(data, cam)
                frames.append(r.render())
            if float(s.done) > 0.5:
                break
    try:
        import imageio.v2 as iio
        iio.mimwrite(path, frames, fps=30, quality=8)
    except Exception:
        from PIL import Image
        path = os.path.splitext(path)[0] + ".gif"
        ims = [Image.fromarray(f).resize((size[1] // 2, size[0] // 2)) for f in frames[::2]]
        ims[0].save(path, save_all=True, append_images=ims[1:], duration=66, loop=0)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--video", type=int, default=4)
    ap.add_argument("--distance", type=float, help="throw: fix the board distance (m)")
    ap.add_argument("--no-dr", action="store_true", help="turn off domain randomisation")
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.run, "config.json")))
    kw = dict(cfg.get("env", {}))
    if a.no_dr:
        kw["dr"] = DR(enabled=False)
    if a.distance and cfg["task"] == "throw":
        kw["distance_range"] = (a.distance, a.distance)
    from train import build_env
    env = build_env(cfg["task"], kw)
    policy, _ = load_policy(a.run, env)
    stats = evaluate(env, policy, a.episodes)
    print(json.dumps(stats, indent=1))
    json.dump(stats, open(os.path.join(a.run, "eval.json"), "w"), indent=1)
    if a.video:
        print("video:", render(env, policy, os.path.join(a.run, "rollout.mp4"), a.video))


if __name__ == "__main__":
    main()
