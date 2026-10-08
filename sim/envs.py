"""GPU-parallel training environments for Capstan Arm 5 (MuJoCo MJX + JAX).

One code path for everything:
  * Brax trainers (PPO, SAC) take these envs directly: thousands in parallel on the GPU.
  * sim/sb3_vec.py wraps the same envs as a Stable-Baselines3 VecEnv.
  * sim/play.py renders a trained policy with CPU MuJoCo (same model file).

Tasks (make_env(name, **kw)):
  reach    put the tool point on a random target, with a random approach pitch
  lift     grasp the beanbag from the floor and lift it 15 cm
  place    pick the bag up and put it down on a target spot
  throw    cornhole: throw the bag, held in the gripper, at a board (random
           distance and direction); 3 points in the hole, 1 on the board

Control (all tasks, 100 Hz): the policy outputs 6 numbers in [-1, 1].
  a[0:5]  change of each joint's position target this step (scaled to the
          motor's no-load speed); the env feeds that velocity forward to a
          moteus-style PD loop running at 500 Hz through the motors'
          torque-speed limits (sim/control.py).  This is the same interface the
          real arm uses: position + velocity setpoints to each moteus.
  a[5]    gripper: -1 = closed, +1 = fully open (servo with delay and rate limit).

Domain randomisation each episode: PD gains, motor torque, servo delay, bag
mass and friction, sensor noise.  So the policy doesn't overfit the model.
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from brax.envs.base import Env, State
from mujoco import mjx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import control as C                                   # noqa: E402
import pinch as P                                     # noqa: E402
from scene import scene_xml, Bag, Board, PINCH_BODIES   # noqa: E402

CTRL_DT = 0.01                                         # default policy rate 100 Hz (throw: 250 Hz)
SIM_DT = 0.002                                         # physics and PD loop 500 Hz
QD_LIM = jnp.array(C.QD_MAX * 2.0)                     # max target speed (rad/s): targets may lead the arm (the motor model limits real speed)
QLO, QHI = jnp.array(C.QLO), jnp.array(C.QHI)
QMID, QHALF = (QLO + QHI) / 2, (QHI - QLO) / 2
G_MAX = C.G_MAX
SERVO_RATE = math.radians(C.SERVO["speed_dps"]) * SIM_DT
MAX_LAT = 8                                            # servo delay buffer, control steps
COUPLED = (("Gl", -1.0), ("Gku", -1.0), ("Gkl", 1.0), ("Gfu", 1.0), ("Gfl", -1.0))
HOME = jnp.radians(jnp.array([0.0, 75.0, -100.0, -45.0, 0.0]))


@dataclass
class DR:
    """Domain randomisation ranges (uniform)."""
    kp: tuple = (0.85, 1.15)
    torque: tuple = (0.85, 1.05)
    servo_lat: tuple = (0.010, 0.025)       # s, command -> servo starts moving
    bag_mass: tuple = (0.8, 1.25)           # x nominal
    bag_friction: tuple = (0.6, 1.0)
    q_noise: float = 0.002                  # rad
    qd_noise: float = 0.02                  # rad/s
    obj_noise: float = 0.003                # m (object position, e.g. from a camera)
    enabled: bool = True


def _u(rng, lo_hi):
    return jax.random.uniform(rng, (), minval=lo_hi[0], maxval=lo_hi[1])


# ============================================================== base env =====
class ArmEnv(Env):
    task_obs_size = 0
    episode_length = 300

    ctrl_dt = CTRL_DT

    def __init__(self, bag: Bag | None = Bag(), board: Board | None = None, dr: DR = DR(),
                 target_site=False, heat_penalty=2e-4, action_rate_penalty=0.02, ctrl_dt=None, impl="jax",
                 gravity_comp=True):
        self.bag_cfg, self.board_cfg, self.dr = bag, board, dr
        self.ctrl_dt = ctrl_dt or self.ctrl_dt
        self.n_sub = int(round(self.ctrl_dt / SIM_DT))
        assert abs(self.n_sub * SIM_DT - self.ctrl_dt) < 1e-9, \
            f"control period {self.ctrl_dt} s must be a whole number of {SIM_DT} s physics steps"
        self.dq_step = QD_LIM * self.ctrl_dt
        self.heat_penalty, self.action_rate_penalty = heat_penalty, action_rate_penalty
        self.xml = scene_xml(bag=bag, board=board, target_site=target_site)
        self.mj_model = mujoco.MjModel.from_xml_string(self.xml)
        self.mj_model.opt.timestep = SIM_DT
        # impl="warp" uses MuJoCo Warp on NVIDIA GPUs (pip install warp-lang mujoco-warp); "jax" works everywhere
        self.mx = mjx.put_model(self.mj_model, impl=impl)
        m = self.mj_model
        jid = lambda n: m.joint(n).id
        self.qadr = jnp.array([m.jnt_qposadr[jid(j)] for j in C.JOINTS])
        self.dadr = jnp.array([m.jnt_dofadr[jid(j)] for j in C.JOINTS])
        self.gq = int(m.jnt_qposadr[jid("G")])
        self.gd = int(m.jnt_dofadr[jid("G")])
        self.coupled = [(int(m.jnt_qposadr[jid(n)]), s) for n, s in COUPLED]
        self.act = jnp.array([m.actuator(j).id for j in C.JOINTS])
        self.gact = int(m.actuator("G").id)
        self.pad_u, self.pad_l = m.site("pad_u").id, m.site("pad_l").id
        self.grip_body = m.body("grip").id
        self.floor = m.geom("floor").id
        geom_body = np.array(m.geom_bodyid)
        arm_bodies = [m.body(n).id for n in ("upper", "fore", "hand", "grip", "l1u", "l1l", "fu", "fl")]
        self.arm_geoms = jnp.array([g for g in range(m.ngeom) if geom_body[g] in arm_bodies and m.geom_contype[g]])
        self.pad_geoms = jnp.array([m.geom("pad_fu").id, m.geom("pad_fl").id])
        self.gravity_comp = gravity_comp
        if bag is not None:
            self.bag_body = m.body("bag").id                  # centre bead (floppy) or the box (rigid)
            self.bag_bodies = jnp.array([b for b in range(m.nbody) if m.body(b).name.startswith("bag")])
            self.bag_geoms = jnp.array([g for g in range(m.ngeom) if geom_body[g] in np.asarray(self.bag_bodies)
                                        and m.geom_contype[g]])
            self.bag_geom = m.geom("bag").id
            self.bq = int(m.jnt_qposadr[jid("bag")])
            self.bd = int(m.jnt_dofadr[jid("bag")])
            self.edge_bodies = jnp.array([m.body(n).id for n in ("bag_px", "bag_nx", "bag_py", "bag_ny")]) \
                if bag.model == "floppy" else jnp.array([self.bag_body] * 4)
            # pinch latch (sim/pinch.py): one weld per bead
            pb = PINCH_BODIES[:bag.n_tiles]
            self.pinch_bodies = jnp.array([m.body(b).id for b in pb])
            self.pinch_eq = jnp.array([m.equality(f"pinch_{b}").id for b in pb])
            self.bag_mass = float(np.sum(m.body_mass[np.asarray(self.bag_bodies)]))
            self._dr_mass0 = jnp.array(m.body_mass)[self.bag_bodies]
            self._dr_inert0 = jnp.array(m.body_inertia)[self.bag_bodies]

    # ------------------------------------------------------------ Brax API --
    @property
    def observation_size(self):
        return 6 + 6 + 3 + 3 + 3 + 6 + self.task_obs_size

    @property
    def action_size(self):
        return 6

    @property
    def backend(self):
        return "mjx"

    # -------------------------------------------------------------- helpers --
    def set_arm(self, qpos, q5, phi):
        qpos = qpos.at[self.qadr].set(q5).at[self.gq].set(phi)
        for adr, s in self.coupled:
            qpos = qpos.at[adr].set(s * phi)
        return qpos

    def model_for(self, info):
        """Physics model with this episode's randomised bag."""
        if self.bag_cfg is None:
            return self.mx
        k = info["dr_bag_mass"]
        row = self.pinch_eq[jnp.maximum(info["pinch_idx"], 0)]
        mx = self.mx.replace(
            eq_data=self.mx.eq_data.at[row].set(info["pinch_data"]),
            body_mass=self.mx.body_mass.at[self.bag_bodies].set(self._dr_mass0 * k),
            body_inertia=self.mx.body_inertia.at[self.bag_bodies].set(self._dr_inert0 * k),
            geom_friction=self.mx.geom_friction.at[self.bag_geoms, 0].set(info["dr_bag_friction"]))
        return mx

    def tcp(self, d):
        return (d.site_xpos[self.pad_u] + d.site_xpos[self.pad_l]) / 2

    def grip_axes(self, d):
        R = d.xmat[self.grip_body]
        return R[:, 0], R[:, 2]          # approach, jaw axis

    def touching(self, d, A, B):
        """True if any geom in A touches any geom in B."""
        c = d._impl.contact
        a, b = c.geom[:, 0], c.geom[:, 1]
        hit = (jnp.isin(a, A) & jnp.isin(b, B)) | (jnp.isin(b, A) & jnp.isin(a, B))
        return jnp.any(hit & (c.dist < 0))

    def pad_grasp(self, d):
        """Both Fin Ray pads touching the bag."""
        return self.touching(d, self.pad_geoms[:1], self.bag_geoms) & self.touching(d, self.pad_geoms[1:], self.bag_geoms)

    def bag_in_grip(self, d):
        """Bag centre and edge beads in the gripper frame (what a wrist camera would see)."""
        R = d.xmat[self.grip_body]
        pts = jnp.concatenate([d.xpos[self.bag_body][None], d.xpos[self.edge_bodies]])
        return ((pts - self.tcp(d)) @ R).ravel()

    def _eq_active(self, d, idx):
        on = jnp.arange(self.pinch_eq.shape[0]) == idx
        return d.eq_active.at[self.pinch_eq].set(on)

    def _pinch_update(self, d, info, g_target):
        """Grab / release / slip of the pinch latch, once per control step."""
        phi = d.qpos[self.gq]
        idx = info["pinch_idx"]
        held = idx >= 0
        tcp = self.tcp(d)
        R = d.xmat[self.grip_body]
        gp, gq = d.xpos[self.grip_body], d.xquat[self.grip_body]
        k = jnp.argmin(jnp.linalg.norm(d.xpos[self.pinch_bodies] - tcp, axis=1))
        b = self.pinch_bodies[k]
        grab = (~held) & (g_target < phi - math.radians(1.0)) & self.pad_grasp(d)
        new_data = P.relpose(jnp, gp, R, gq, d.xpos[b], d.xquat[b])
        hb = self.pinch_bodies[jnp.maximum(idx, 0)]
        p_now = d.xpos[hb]                                       # the pinched bead (not the floppy centre)
        v = (p_now - info["bag_p_prev"]) / self.ctrl_dt
        phi_ref = jnp.where(held, jnp.minimum(info["phi_pinch"], phi), info["phi_pinch"])   # tightest squeeze
        acc = (v - info["bag_v_prev"]) / self.ctrl_dt
        need = self.bag_mass * info["dr_bag_mass"] * jnp.linalg.norm(acc - jnp.array([0.0, 0.0, -9.81]))
        slip = held & (info["held_steps"] >= 2) & (need > P.hold_force(jnp, phi, C.SERVO["stall_nm"]))
        opened = held & (phi > phi_ref + math.radians(P.RELEASE_DEG))
        idx = jnp.where(grab, k, idx)
        idx = jnp.where(slip | opened, -1, idx).astype(jnp.int32)
        info.update(pinch_idx=idx,
                    pinch_data=jnp.where(grab, new_data, info["pinch_data"]),
                    phi_pinch=jnp.where(grab, phi, phi_ref),
                    bag_v_prev=jnp.where(idx >= 0, v, jnp.zeros(3)),
                    bag_p_prev=d.xpos[self.pinch_bodies[jnp.maximum(idx, 0)]],
                    held_steps=jnp.where((idx >= 0) & ~grab, info["held_steps"] + 1, 0.0),
                    slipped=jnp.maximum(info["slipped"], slip.astype(float)))
        return d.replace(eq_active=self._eq_active(d, idx)), info

    def arm_hits_floor(self, d):
        """Lowest corner of any arm collision box below the floor (no contact solve needed)."""
        g = self.arm_geoms
        R = d.geom_xmat[g]                                   # (n, 3, 3)
        reach = jnp.sum(jnp.abs(R[:, 2, :]) * self.mx.geom_size[g], axis=1)
        return jnp.any(d.geom_xpos[g, 2] - reach < -0.002)

    # ---------------------------------------------------------- reset/step --
    def reset(self, rng):
        rng, r_dr, r_task = jax.random.split(rng, 3)
        info = self._sample_dr(r_dr)
        if self.bag_cfg is not None:
            info.update(pinch_idx=jnp.asarray(-1, jnp.int32), pinch_data=jnp.zeros(11), phi_pinch=jnp.zeros(()))
        qpos = jnp.array(self.mj_model.qpos0)
        qpos = self.set_arm(qpos, HOME, 0.3)
        qpos, info = self.task_reset(r_task, qpos, info)
        d = mjx.make_data(self.mx).replace(qpos=qpos)
        d = mjx.forward(self.model_for(info), d)
        qpos, info = self.task_post_forward(d, qpos, info)
        d = mjx.forward(self.model_for(info), d.replace(qpos=qpos))
        info.update(q_tgt=d.qpos[self.qadr], g_cmd=d.qpos[self.gq], g_queue=jnp.full(MAX_LAT, d.qpos[self.gq]),
                    last_act=jnp.zeros(6), rng=rng, t=jnp.zeros(()))
        if self.bag_cfg is not None:
            info.update(bag_v_prev=jnp.zeros(3), slipped=jnp.zeros(()), held_steps=jnp.zeros(()),
                        bag_p_prev=d.xpos[self.pinch_bodies[jnp.maximum(info["pinch_idx"], 0)]])
            d = d.replace(eq_active=self._eq_active(d, info["pinch_idx"]))
        info = self.task_init_info(d, info)
        obs = self._obs(d, info)
        metrics = {k: jnp.zeros((), jnp.float32) for k in self.metric_keys()}
        strong = lambda x: jnp.asarray(x) + jnp.zeros((), jnp.asarray(x).dtype)
        info = jax.tree.map(strong, info)
        return State(d, obs, jnp.zeros(()), jnp.zeros(()), metrics, info)

    def _sample_dr(self, rng):
        r = jax.random.split(rng, 6)
        on = self.dr.enabled
        f = lambda rr, rg, nominal: _u(rr, rg) if on else jnp.asarray(nominal, float)
        return dict(dr_kp=f(r[0], self.dr.kp, 1.0), dr_torque=f(r[1], self.dr.torque, 1.0),
                    dr_lat=jnp.round((_u(r[2], self.dr.servo_lat) if on else jnp.asarray(0.015)) / self.ctrl_dt
                                     ).astype(int),
                    dr_bag_mass=f(r[3], self.dr.bag_mass, 1.0), dr_bag_friction=f(r[4], self.dr.bag_friction,
                                                                                  self.bag_cfg.friction if self.bag_cfg else 0.8))

    def step(self, state, action):
        info = dict(state.info)
        a = jnp.clip(action, -1.0, 1.0)
        mx = self.model_for(info)
        q_tgt_new = jnp.clip(info["q_tgt"] + a[:5] * self.dq_step, QLO, QHI)
        qd_ff = (q_tgt_new - info["q_tgt"]) / self.ctrl_dt
        g_want = (a[5] + 1.0) / 2.0 * G_MAX
        queue = jnp.roll(info["g_queue"], 1).at[0].set(g_want)
        g_target = queue[jnp.clip(info["dr_lat"], 0, MAX_LAT - 1)]

        def sub(carry, i):
            d, g_cmd, heat = carry
            s = (i + 1.0) / self.n_sub
            q_t = info["q_tgt"] + (q_tgt_new - info["q_tgt"]) * s
            q, qd = d.qpos[self.qadr], d.qvel[self.dadr]
            ff = jnp.where(self.gravity_comp, d.qfrc_bias[self.dadr] - jnp.array([0.0, 1.0, 0, 0, 0]) * C.balancer(jnp, q[1]), 0.0)
            tau, cur = C.joint_torque(jnp, q, qd, q_t, qd_ff, info["dr_kp"], info["dr_torque"], tau_ff=ff)
            g_cmd = g_cmd + jnp.clip(g_target - g_cmd, -SERVO_RATE, SERVO_RATE)
            ctrl = d.ctrl.at[self.act].set(tau).at[self.gact].set(g_cmd)
            d = mjx.step(mx, d.replace(ctrl=ctrl))
            return (d, g_cmd, heat + jnp.sum(cur * cur * C.RES)), None

        (d, g_cmd, heat), _ = jax.lax.scan(sub, (state.pipeline_state, info["g_cmd"], 0.0), jnp.arange(self.n_sub))
        info.update(q_tgt=q_tgt_new, g_cmd=g_cmd, g_queue=queue, t=info["t"] + self.ctrl_dt)
        if self.bag_cfg is not None:
            d, info = self._pinch_update(d, info, g_target)
        reward, done, info, metrics = self.task_reward(d, info, a)
        heat_w = heat / self.n_sub
        reward = reward - self.heat_penalty * heat_w - self.action_rate_penalty * jnp.sum((a - info["last_act"]) ** 2)
        info["last_act"] = a
        metrics = dict(metrics, heat_w=heat_w)
        bad = jnp.any(jnp.isnan(d.qpos)) | jnp.any(jnp.abs(d.qvel) > 200)
        done = jnp.where(bad, 1.0, done)
        reward = jnp.where(bad, -10.0, reward)
        obs = self._obs(d, info)
        strong = lambda x: jnp.asarray(x) + jnp.zeros((), jnp.asarray(x).dtype)    # drop weak types (avoids a retrace)
        info = jax.tree.map(strong, info)
        metrics = jax.tree.map(strong, metrics)
        return state.replace(pipeline_state=d, obs=obs, reward=reward, done=done.astype(float),
                             metrics=dict(state.metrics, **metrics), info=info)

    def _obs(self, d, info):
        rng = jax.random.fold_in(info["rng"], jnp.round(info["t"] / self.ctrl_dt).astype(int))
        n = jax.random.normal(rng, (12,))
        q = jnp.concatenate([d.qpos[self.qadr], d.qpos[self.gq][None]])
        qd = jnp.concatenate([d.qvel[self.dadr], d.qvel[self.gd][None]])
        q = q + n[:6] * self.dr.q_noise
        qd = qd + n[6:] * self.dr.qd_noise
        qn = (q[:5] - QMID) / QHALF
        x, z = self.grip_axes(d)
        base = jnp.concatenate([qn, q[5:] / G_MAX, qd / 10.0, self.tcp(d), x, z, info["last_act"]])
        return jnp.concatenate([base, self.task_obs(d, info)])

    # ---------------------------------------------------------- task hooks --
    def task_reset(self, rng, qpos, info):
        return qpos, info

    def task_post_forward(self, d, qpos, info):
        return qpos, info

    def task_init_info(self, d, info):
        return info

    def task_obs(self, d, info):
        return jnp.zeros(0)

    def task_reward(self, d, info, a):
        return jnp.zeros(()), jnp.zeros(()), info, {}

    def metric_keys(self):
        return ("heat_w",)


# ================================================================== reach ====
class Reach(ArmEnv):
    """Tool point to a target with a given approach pitch.  Targets come from
    random reachable joint configurations, so every one can be reached."""
    task_obs_size = 3 + 1
    episode_length = 300

    def __init__(self, **kw):
        kw.setdefault("bag", None)
        super().__init__(target_site=True, **kw)

    def task_reset(self, rng, qpos, info):
        r1, r2 = jax.random.split(rng)
        q = QLO + (QHI - QLO) * jax.random.uniform(r1, (5,), minval=0.1, maxval=0.9)
        q = q.at[1].set(jax.random.uniform(r2, (), minval=math.radians(20), maxval=math.radians(110)))
        info["goal_q"] = q
        start = HOME + jax.random.uniform(r2, (5,), minval=-0.3, maxval=0.3)
        return self.set_arm(qpos, jnp.clip(start, QLO, QHI), 0.3), info

    def task_init_info(self, d, info):
        # forward kinematics of the goal configuration -> target position and approach
        dg = mjx.kinematics(self.mx, d.replace(qpos=self.set_arm(d.qpos, info["goal_q"], 0.3)))
        info["goal"] = self.tcp(dg)
        info["goal_x"] = self.grip_axes(dg)[0]
        info["goal_pitch"] = jnp.arcsin(jnp.clip(info["goal_x"][2], -1, 1))
        return info

    def task_obs(self, d, info):
        return jnp.concatenate([info["goal"] - self.tcp(d), info["goal_pitch"][None]])

    def task_reward(self, d, info, a):
        dist = jnp.linalg.norm(info["goal"] - self.tcp(d))
        align = jnp.dot(self.grip_axes(d)[0], info["goal_x"])
        r = (1 - jnp.tanh(dist / 0.10)) + (1 - jnp.tanh(dist / 0.01)) + 0.5 * (align + 1) / 2
        floor = self.arm_hits_floor(d)
        r = r - 1.0 * floor
        return r, jnp.zeros(()), info, dict(dist=dist, align=align, floor_hit=floor.astype(float))

    def metric_keys(self):
        return ("heat_w", "dist", "align", "floor_hit")


# =================================================================== lift ====
class Lift(ArmEnv):
    """Grasp the bag from the floor and lift it to 15 cm."""
    task_obs_size = 3 + 3 + 4 + 1 + 12 + 1
    episode_length = 400
    lift_z = 0.15

    def __init__(self, bag=Bag(), **kw):
        super().__init__(bag=bag, **kw)

    def task_reset(self, rng, qpos, info):
        r = jax.random.split(rng, 3)
        rad = jax.random.uniform(r[0], (), minval=0.20, maxval=0.36)
        ang = jax.random.uniform(r[1], (), minval=-1.0, maxval=1.0)
        yaw = jax.random.uniform(r[2], (), minval=-math.pi, maxval=math.pi)
        p = jnp.array([rad * jnp.cos(ang), rad * jnp.sin(ang), self.bag_cfg.thick / 2 + 0.001])
        q = jnp.array([jnp.cos(yaw / 2), 0.0, 0.0, jnp.sin(yaw / 2)])
        qpos = qpos.at[self.bq:self.bq + 3].set(p).at[self.bq + 3:self.bq + 7].set(q)
        return qpos, info

    def bag_obs(self, d, info):
        rng = jax.random.fold_in(info["rng"], 7)
        bp = d.xpos[self.bag_body] + jax.random.normal(rng, (3,)) * self.dr.obj_noise
        return bp, d.xquat[self.bag_body]

    def task_obs(self, d, info):
        bp, bq = self.bag_obs(d, info)
        edges = (d.xpos[self.edge_bodies] - self.tcp(d)).ravel()       # how the floppy bag lies / hangs
        held = (info["pinch_idx"] >= 0).astype(float)[None]      # the servo can tell it stalled on something
        return jnp.concatenate([bp - self.tcp(d), bp, bq, self.pad_grasp(d).astype(float)[None], edges, held])

    def lift_terms(self, d, info=None):
        bp = d.xpos[self.bag_body]
        reach = jnp.linalg.norm(bp - self.tcp(d))
        grasp = self.pad_grasp(d) if info is None else (info["pinch_idx"] >= 0)
        h = bp[2] - self.bag_cfg.thick / 2
        lifted = jnp.clip(h / self.lift_z, 0, 1)
        return reach, grasp, h, lifted

    def task_reward(self, d, info, a):
        reach, grasp, h, lifted = self.lift_terms(d, info)
        r = (1 - jnp.tanh(reach / 0.05)) + 0.5 * grasp + 4.0 * lifted * grasp
        r = r - 0.5 * self.arm_hits_floor(d)
        success = (h > self.lift_z * 0.9) & grasp
        return r, jnp.zeros(()), info, dict(reach=reach, grasp=grasp.astype(float), height=h,
                                            success=success.astype(float))

    def metric_keys(self):
        return ("heat_w", "reach", "grasp", "height", "success")


# ================================================================== place ====
class Place(Lift):
    """Pick the bag up and put it down on a target spot."""
    task_obs_size = Lift.task_obs_size + 3
    episode_length = 500

    def task_reset(self, rng, qpos, info):
        r1, r2 = jax.random.split(rng)
        qpos, info = super().task_reset(r1, qpos, info)
        a, b = jax.random.split(r2)
        rad = jax.random.uniform(a, (), minval=0.20, maxval=0.36)
        ang = jax.random.uniform(b, (), minval=-1.2, maxval=1.2)
        info["place"] = jnp.array([rad * jnp.cos(ang), rad * jnp.sin(ang), self.bag_cfg.thick / 2])
        return qpos, info

    def task_init_info(self, d, info):
        info["carried"] = jnp.zeros(())
        return info

    def task_obs(self, d, info):
        return jnp.concatenate([super().task_obs(d, info), info["place"] - d.xpos[self.bag_body]])

    def task_reward(self, d, info, a):
        reach, grasp, h, lifted = self.lift_terms(d, info)
        bp = d.xpos[self.bag_body]
        to_goal = jnp.linalg.norm((info["place"] - bp)[:2])
        carried = grasp & (h > 0.03)
        was_carried = jnp.maximum(info["carried"], carried.astype(float)) > 0
        info["carried"] = was_carried.astype(float)
        on_spot = (to_goal < 0.025) & (h < 0.01) & was_carried
        released = on_spot & ~grasp
        r = ((1 - jnp.tanh(reach / 0.05)) * (1 - on_spot) + 0.5 * grasp + 1.0 * carried
             + 3.0 * carried * (1 - jnp.tanh(to_goal / 0.08)) + 5.0 * on_spot + 3.0 * released)
        r = r - 0.5 * self.arm_hits_floor(d)
        return r, jnp.zeros(()), info, dict(reach=reach, grasp=grasp.astype(float), to_goal=to_goal,
                                            success=released.astype(float))

    def metric_keys(self):
        return ("heat_w", "reach", "grasp", "to_goal", "success")


# ================================================================== throw ====
class Throw(ArmEnv):
    """Cornhole.  The bag starts in the gripper at a random ready pose; the
    board is at a random distance and direction.  Points when the bag comes
    to rest: 3 in the hole, 1 on the board.  Dense shaping at release from
    the bag's ballistic landing point, and at landing from where it hit."""
    task_obs_size = 15 + 3 + 3 + 3 + 2 + 1
    episode_length = 625         # 2.5 s at 250 Hz
    ctrl_dt = 0.004              # 250 Hz (2 physics steps): release timing matters to a few ms

    # Default: 1/3-scale board (8 x 16 in, 2 in hole; regulation pitch scales to 2.74 m) and a
    # floppy 2 in, 25 g bag.  Every episode starts from a real table pickup out of the grasp
    # library (sim/make_grasps.py), so the policy learns to throw from whatever grip it got.
    def __init__(self, bag=Bag(), board=Board(scale=1 / 3, distance=2.0),
                 distance_range=(1.0, 2.4), yaw_range=(-30.0, 30.0), grasps="default", **kw):
        self.distance_range, self.yaw_range = distance_range, yaw_range
        super().__init__(bag=bag, board=board, **kw)
        self.grasps = None
        path = os.path.join(HERE, "assets", "grasps.npz") if grasps == "default" else grasps
        if path and os.path.exists(path) and bag.model == "floppy":
            g = np.load(path)
            if g["qpos"].shape[1] == self.mj_model.nq and "pinch" in g:
                self.grasps = dict(qpos=jnp.array(g["qpos"]), q_tgt=jnp.array(g["q_tgt"]),
                                   pinch=jnp.array(g["pinch"][:, 0], jnp.int32), pinch_data=jnp.array(g["pinch_data"]),
                                   phi_pinch=jnp.array(g["phi_pinch"][:, 0]))
        if self.grasps is None and type(self) is Throw:
            print("throw: no grasp library for this bag; starting with the bag placed in the gripper "
                  "(run python sim/make_grasps.py)")
        m = self.mj_model
        self.board_body = m.body("board").id
        self.board_geoms = jnp.array([g for g in range(m.ngeom) if m.geom_bodyid[g] == self.board_body]
                                     + [m.geom("board_skirt").id])
        self.board_pos0 = jnp.array(m.body_pos[self.board_body])
        self.skirt_geom = m.geom("board_skirt").id
        self.skirt_pos0 = jnp.array(m.geom_pos[self.skirt_geom])
        b = board
        self.hole_local = jnp.array([b.length - 9 * 0.0254 * b.scale, 0.0, 0.0])   # in the board's top frame
        self.hole_r = b.hole_d / 2
        self.board_wl = jnp.array([b.length, b.width])
        self.world_geoms = jnp.concatenate([jnp.array([self.floor]), self.board_geoms])

    def model_for(self, info):
        mx = super().model_for(info)
        # move the board for this episode (static body: change its pose in the model)
        yaw, dist = info["board_yaw"], info["board_dist"]
        c, s = jnp.cos(yaw), jnp.sin(yaw)
        pos = jnp.array([dist * c, dist * s, self.board_pos0[2]])
        tilt = -self.board_cfg.tilt
        qy = jnp.array([jnp.cos(tilt / 2), 0.0, jnp.sin(tilt / 2), 0.0])
        qz = jnp.array([jnp.cos(yaw / 2), 0.0, 0.0, jnp.sin(yaw / 2)])
        quat = _qmul(qz, qy)
        skirt = jnp.array([dist * c, dist * s, self.skirt_pos0[2]])
        return mx.replace(body_pos=mx.body_pos.at[self.board_body].set(pos),
                          body_quat=mx.body_quat.at[self.board_body].set(quat),
                          geom_pos=mx.geom_pos.at[self.skirt_geom].set(skirt),
                          geom_quat=mx.geom_quat.at[self.skirt_geom].set(qz))

    def task_reset(self, rng, qpos, info):
        r = jax.random.split(rng, 4)
        info["board_dist"] = jax.random.uniform(r[0], (), minval=self.distance_range[0], maxval=self.distance_range[1])
        info["board_yaw"] = jnp.radians(jax.random.uniform(r[1], (), minval=self.yaw_range[0], maxval=self.yaw_range[1]))
        if self.grasps is not None:
            i = jax.random.randint(r[3], (), 0, self.grasps["qpos"].shape[0])
            info["phi_hold"] = jnp.zeros(())                      # servo commanded shut; the bag stops it
            info["from_library"] = jnp.ones(())
            info.update(pinch_idx=self.grasps["pinch"][i], pinch_data=self.grasps["pinch_data"][i],
                        phi_pinch=self.grasps["phi_pinch"][i])
            return self.grasps["qpos"][i], info
        info["from_library"] = jnp.zeros(())
        ready = jnp.radians(jnp.array([0.0, 100.0, -70.0, -40.0, 0.0]))
        ready = ready + jax.random.uniform(r[2], (5,), minval=-0.15, maxval=0.15)
        phi = jnp.arcsin(jnp.clip((self.bag_cfg.thick * 1000 / 2 + C.META["grip"]["FACE"] - C.META["grip"]["PZ"])
                                  / C.META["grip"]["LL"], -1, 1))
        info["phi_hold"] = phi - math.radians(6)
        return self.set_arm(qpos, jnp.clip(ready, QLO, QHI), phi), info

    def task_post_forward(self, d, qpos, info):
        if self.grasps is not None:
            return qpos, info
        mid = self.tcp(d)
        R = d.xmat[self.grip_body]
        c = mid + R[:, 0] * max(0.0, self.bag_cfg.size / 2 - 0.024)
        quat = _mat2quat(R)
        return qpos.at[self.bq:self.bq + 3].set(c).at[self.bq + 3:self.bq + 7].set(quat), info

    def task_init_info(self, d, info):
        g0 = info["phi_hold"] / G_MAX * 2 - 1
        info.update(g_cmd=info["phi_hold"], g_queue=jnp.full(MAX_LAT, info["phi_hold"]),
                    last_act=jnp.zeros(6).at[5].set(g0),
                    released=jnp.zeros(()), landed=jnp.zeros(()), land_err=jnp.full((), 5.0, jnp.float32).astype(jnp.float32) + jnp.zeros((), jnp.float32),
                    rest_steps=jnp.zeros(()), scored=jnp.zeros(()), through=jnp.zeros(()))
        mx = self.model_for(info)
        info["hole"] = self._hole_world(mx)
        return info

    def _hole_world(self, mx):
        pos, quat = mx.body_pos[self.board_body], mx.body_quat[self.board_body]
        return pos + _qrot(quat, self.hole_local)

    def _in_board_frame(self, mx, p):
        pos, quat = mx.body_pos[self.board_body], mx.body_quat[self.board_body]
        return _qrot(_qconj(quat), p - pos)

    def task_obs(self, d, info):
        bv = d.qvel[self.bd:self.bd + 3]
        h = info["hole"]
        rel = h - self.tcp(d)
        polar = jnp.array([jnp.hypot(h[0], h[1]), jnp.arctan2(h[1], h[0])])
        noise = jax.random.normal(jax.random.fold_in(info["rng"], 11), (15,)) * self.dr.obj_noise
        held = (info["pinch_idx"] >= 0).astype(float)[None]
        return jnp.concatenate([self.bag_in_grip(d) + noise, bv / 5.0, rel, h, polar, held])

    def task_reward(self, d, info, a):
        return self.throw_reward(d, info, armed=True)

    def throw_reward(self, d, info, armed):
        """Cornhole scoring.  armed=False (before a pickup has happened) disables
        release / drop detection."""
        mx = self.model_for(info)
        bp = d.xpos[self.bag_body]
        bv = d.qvel[self.bd:self.bd + 3]
        mid = self.tcp(d)
        just_released = armed & (info["released"] == 0) & (jnp.linalg.norm(bp - mid) > 0.04 + self.bag_cfg.size / 2)
        # predicted landing on the board plane height from the release state (ballistic)
        hz = info["hole"][2]
        disc = bv[2] ** 2 + 2 * 9.81 * jnp.maximum(bp[2] - hz, 0.0)
        tf = (bv[2] + jnp.sqrt(jnp.maximum(disc, 0))) / 9.81
        pred = bp[:2] + bv[:2] * tf
        pred_err = jnp.linalg.norm(pred - info["hole"][:2])
        r_release = just_released * 2.0 * jnp.exp(-(pred_err / 0.25) ** 2)
        released = jnp.maximum(info["released"], just_released.astype(float))
        # landing = first contact with the floor or the board after release
        impact = (released > 0) & (info["landed"] == 0) & self.touching(d, self.bag_geoms, self.world_geoms)
        land_err = jnp.where(impact, jnp.linalg.norm(bp[:2] - info["hole"][:2]), info["land_err"])
        r_land = impact * 3.0 * jnp.exp(-(land_err / 0.15) ** 2)
        landed = jnp.maximum(info["landed"], impact.astype(float))
        # through the hole: bag centre passes below the top surface inside the hole
        pb = self._in_board_frame(mx, bp)
        hs = self.hole_r * math.sqrt(math.pi) / 2                  # the modelled hole: equal-area square (half side)
        thru_now = (jnp.abs(pb[0] - self.hole_local[0]) < hs) & (jnp.abs(pb[1]) < hs) & (pb[2] < -0.005) & (released > 0)
        through = jnp.maximum(info["through"], thru_now.astype(float))
        # otherwise score where it comes to rest
        still = (landed > 0) & (jnp.linalg.norm(bv) < 0.05)
        rest_steps = jnp.where(still, info["rest_steps"] + 1, 0.0)
        on_board = (pb[0] > 0) & (pb[0] < self.board_wl[0]) & (jnp.abs(pb[1]) < self.board_wl[1] / 2) & \
                   (pb[2] > -2 * self.bag_cfg.thick) & (pb[2] < self.bag_cfg.thick * 1.5)   # incl. hanging in the hole edge
        in_hole = through > 0
        land_err = jnp.where(thru_now & (info["through"] == 0), 0.0, land_err)
        points = jnp.where(in_hole, 3.0, jnp.where(on_board, 1.0, 0.0))
        settle = (rest_steps >= 10) | in_hole
        final = settle & (info["scored"] == 0)
        r_final = final * 5.0 * points
        dropped = armed & (released == 0) & (bp[2] < 0.03)    # dropped before throwing
        off = jnp.linalg.norm(bp[:2]) > 6.0
        done = (final | dropped | off).astype(float)
        r = r_release + r_land + r_final - 3.0 * (dropped & (info["scored"] == 0)) - 0.3 * self.arm_hits_floor(d)
        info.update(released=released, landed=landed, land_err=land_err, rest_steps=rest_steps,
                    scored=jnp.maximum(info["scored"], final.astype(float)), through=through)
        return r, done, info, dict(points=(final * points).astype(jnp.float32) + jnp.zeros((), jnp.float32), in_hole=(final & in_hole).astype(float),
                                   on_board=(final & on_board & ~in_hole).astype(float),
                                   land_err=jnp.where(landed > 0, land_err, 0.0),
                                   released=released, release_err=jnp.where(just_released, pred_err, 0.0))

    def metric_keys(self):
        return ("heat_w", "points", "in_hole", "on_board", "land_err", "released", "release_err")


# ============================================================= pick+throw ====
class PickThrow(Throw):
    """The whole thing: pick the floppy bag up off the table, then throw it at
    the board.  Pick-up shaping (reach, pinch, lift) until the bag is carried,
    cornhole scoring after.  Harder than throw alone; train throw from the
    grasp library first and use this to fine-tune, or chain a lift policy
    and a throw policy."""
    task_obs_size = Throw.task_obs_size + 3 + 1
    episode_length = 1250        # 5 s at 250 Hz

    def __init__(self, **kw):
        kw.setdefault("grasps", None)
        super().__init__(**kw)
        self.grasps = None

    def task_reset(self, rng, qpos, info):
        r1, r2 = jax.random.split(rng)
        qpos, info = super().task_reset(r1, qpos, info)       # board (no library: bag gets moved below)
        a, b, c = jax.random.split(r2, 3)
        rad = jax.random.uniform(a, (), minval=0.17, maxval=0.27)
        ang = jax.random.uniform(b, (), minval=-1.0, maxval=1.0)
        yaw = jax.random.uniform(c, (), minval=-math.pi, maxval=math.pi)
        qpos = self.set_arm(qpos, HOME, G_MAX * 0.8)
        p = jnp.array([rad * jnp.cos(ang), rad * jnp.sin(ang), self.bag_cfg.thick / 2 + 0.001])
        q = jnp.array([jnp.cos(yaw / 2), 0.0, 0.0, jnp.sin(yaw / 2)])
        qpos = qpos.at[self.bq:self.bq + 3].set(p).at[self.bq + 3:self.bq + 7].set(q)
        info["phi_hold"] = jnp.asarray(G_MAX * 0.8)
        return qpos, info

    def task_post_forward(self, d, qpos, info):
        return qpos, info

    def task_init_info(self, d, info):
        info = super().task_init_info(d, info)
        info["carried"] = jnp.zeros(())
        return info

    def task_obs(self, d, info):
        bp = d.xpos[self.bag_body] + jax.random.normal(jax.random.fold_in(info["rng"], 7), (3,)) * self.dr.obj_noise
        return jnp.concatenate([super().task_obs(d, info), bp, info["carried"][None]])

    def task_reward(self, d, info, a):
        bp = d.xpos[self.bag_body]
        reach = jnp.linalg.norm(bp - self.tcp(d))
        grasp = info["pinch_idx"] >= 0
        h = bp[2] - self.bag_cfg.thick / 2
        now_carried = grasp & (h > 0.05)
        carried = jnp.maximum(info["carried"], now_carried.astype(float))
        first = (carried > 0) & (info["carried"] == 0)
        r_pick = (1 - carried) * ((1 - jnp.tanh(reach / 0.05)) + 0.5 * grasp + 2.0 * jnp.clip(h / 0.05, 0, 1) * grasp) * 0.2
        info["carried"] = carried
        r, done, info, metrics = self.throw_reward(d, info, armed=carried > 0)
        return r + r_pick + 3.0 * first, done, info, dict(metrics, picked=carried)

    def metric_keys(self):
        return Throw.metric_keys(self) + ("picked",)


# --------------------------------------------------------------- quaternions --
def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return jnp.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _qconj(q):
    return q * jnp.array([1.0, -1.0, -1.0, -1.0])


def _qrot(q, v):
    return _qmul(_qmul(q, jnp.concatenate([jnp.zeros(1), v])), _qconj(q))[1:]


def _mat2quat(R):
    # robust enough for rotation matrices from MuJoCo
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    w = jnp.sqrt(jnp.maximum(1 + tr, 1e-9)) / 2
    x = (R[2, 1] - R[1, 2]) / (4 * w)
    y = (R[0, 2] - R[2, 0]) / (4 * w)
    z = (R[1, 0] - R[0, 1]) / (4 * w)
    q = jnp.array([w, x, y, z])
    return q / jnp.linalg.norm(q)


TASKS = {"reach": Reach, "lift": Lift, "place": Place, "throw": Throw, "pick_throw": PickThrow}


def make_env(name, **kw):
    return TASKS[name](**kw)
