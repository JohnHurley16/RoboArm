"""End-to-end check of the throw env: replay the throw study's best open-loop
throw through the env's action interface and see that the env scores it."""
import json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax, jax.numpy as jnp, numpy as np
from envs import make_env, DR, QLO, QHI
import throw_study as T

def windup(q_from, q_to, ctrl_dt):
    """Same wind-up as the CPU controller: minimum-jerk over 0.8 x SETTLE, then hold."""
    n_move, n_hold = int(T.SETTLE * 0.8 / ctrl_dt), int(T.SETTLE * 0.2 / ctrl_dt)
    s = np.linspace(0, 1, n_move)[:, None]
    f = s ** 3 * (10 - 15 * s + 6 * s * s)
    return np.vstack([q_from + (q_to - q_from) * f, np.repeat(q_to[None], n_hold, 0)])


def plan(params, ctrl_dt, n):
    p = np.clip(np.array(params), T.LO, T.HI)
    q0, q1, ts, du, trel = p[0:3], p[3:6], p[6:9], p[9:12], p[12]
    qs, grip = [], []
    for k in range(n):
        t = k * ctrl_dt
        q = np.radians([0.0, *q0, 0.0])
        for i in range(3):
            f, _ = T.minjerk((t - ts[i]) / du[i])
            q[i + 1] = math.radians(q0[i] + (q1[i] - q0[i]) * f)
        qs.append(q); grip.append(1.0 if t >= trel else -1.0)
    return np.array(qs), np.array(grip)

if __name__ == "__main__":
    best = json.load(open(os.path.join(T.OUT, "throw_study.json")))["mini 2 in, 25 g"]["params"]
    dist = float(sys.argv[1]) if len(sys.argv) > 1 else 2.75
    env = make_env("throw", dr=DR(enabled=False), distance_range=(dist, dist), yaw_range=(0, 0))
    s = jax.jit(env.reset)(jax.random.PRNGKey(0))
    step = jax.jit(env.step)
    q0 = np.asarray(s.pipeline_state.qpos[env.qadr])
    qs, grip = plan(best, env.ctrl_dt, env.episode_length)
    # first move from the env's ready pose to the throw's wind-up (0.6 s), then throw
    pre = windup(q0, qs[0], env.ctrl_dt)
    n_pre = len(pre)
    targets = np.vstack([pre, qs])
    gr = np.r_[np.full(n_pre, -1.0), grip]
    q_tgt = q0.copy(); total = 0.0; hold = float(s.info["phi_hold"]) / env.mj_model.jnt_range[env.mj_model.joint("G").id][1] * 2 - 1
    for k in range(env.episode_length):
        dq = (targets[min(k + 1, len(targets) - 1)] - q_tgt) / np.asarray(env.dq_step)
        a = np.r_[np.clip(dq, -1, 1), hold if gr[k] < 0 else 1.0]
        q_tgt = np.clip(q_tgt + a[:5] * np.asarray(env.dq_step), np.asarray(QLO), np.asarray(QHI))
        s = step(s, jnp.asarray(a, jnp.float32))
        total += float(s.reward)
        if float(s.metrics["release_err"]) > 0:
            print(f"t {k*env.ctrl_dt:.3f}s released; predicted miss {float(s.metrics['release_err']):.2f} m")
        if float(s.done) > 0.5:
            break
    bp = np.asarray(s.pipeline_state.xpos[env.bag_body])
    print(f"board at {dist} m, hole {np.round(np.asarray(s.info['hole']),2)}; bag ended at {np.round(bp,2)} after {k*env.ctrl_dt:.2f}s; "
          f"landing miss {float(s.info['land_err']):.2f} m; points {float(s.metrics['points'])}; return {total:.2f}; "
          f"slipped {float(s.info.get('slipped', 0))}")


def record(dist=2.75, path=None, fps=40, size=(540, 960)):
    """Render the replayed throw with the full visual model (GIF)."""
    import mujoco
    from PIL import Image
    from scene import scene_xml
    best = json.load(open(os.path.join(T.OUT, "throw_study.json")))["mini 2 in, 25 g"]["params"]
    env = make_env("throw", dr=DR(enabled=False), distance_range=(dist, dist), yaw_range=(0, 0))
    s = jax.jit(env.reset)(jax.random.PRNGKey(0)); step = jax.jit(env.step)
    full = mujoco.MjModel.from_xml_string(scene_xml(bag=env.bag_cfg, board=env.board_cfg, lean=False))
    mx = env.model_for(s.info)
    full.body_pos[full.body("board").id] = np.asarray(mx.body_pos[env.board_body])
    full.body_quat[full.body("board").id] = np.asarray(mx.body_quat[env.board_body])
    full.geom_pos[full.geom("board_skirt").id] = np.asarray(mx.geom_pos[env.skirt_geom])
    data = mujoco.MjData(full); r = mujoco.Renderer(full, *size)
    cam = mujoco.MjvCamera(); cam.lookat[:] = [dist * 0.55, 0.0, 0.32]; cam.distance = 3.3; cam.azimuth = 90; cam.elevation = -10
    q0 = np.asarray(s.pipeline_state.qpos[env.qadr]); qs, grip = plan(best, env.ctrl_dt, env.episode_length)
    pre = windup(q0, qs[0], env.ctrl_dt); n_pre = len(pre); targets = np.vstack([pre, qs]); gr = np.r_[np.full(n_pre, -1.0), grip]
    hold = float(s.info["phi_hold"]) / float(env.mj_model.jnt_range[env.mj_model.joint("G").id][1]) * 2 - 1
    q_tgt, frames, every = q0.copy(), [], max(1, int(round(1 / fps / env.ctrl_dt)))
    for k in range(env.episode_length):
        dq = (targets[min(k + 1, len(targets) - 1)] - q_tgt) / np.asarray(env.dq_step)
        a = np.r_[np.clip(dq, -1, 1), hold if gr[k] < 0 else 1.0]
        q_tgt = np.clip(q_tgt + a[:5] * np.asarray(env.dq_step), np.asarray(QLO), np.asarray(QHI))
        s = step(s, jnp.asarray(a, jnp.float32))
        if k % every == 0:
            data.qpos[:] = np.asarray(s.pipeline_state.qpos); mujoco.mj_forward(full, data)
            r.update_scene(data, cam); frames.append(Image.fromarray(r.render()))
        if float(s.done) > 0.5:
            for _ in range(fps // 2):
                frames.append(frames[-1])
            break
    path = path or os.path.join(T.OUT, "scripted_throw.gif")
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps * 2), loop=0)
    print("points", float(s.metrics["points"]), "->", path)
    return path
