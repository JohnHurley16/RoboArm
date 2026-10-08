"""Scripted skills on CPU MuJoCo, driven the way the real arm will be:
IK (ik/capstan_ik.py) -> joint position + velocity setpoints -> moteus PD with
gravity feedforward -> motors (sim/control.py).  The same routines can drive
the real arm by swapping the Sim backend for the moteus one.

    sim = Sim(Bag(), bag_pose=(0.22, 0.05, 30))
    ok = pick(sim)                    # pinch the bag off the table and lift it
"""
import math
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "ik"))
import control as C                       # noqa: E402
import pinch as P                         # noqa: E402
import capstan_ik as K                    # noqa: E402
from scene import scene_xml, Bag, Board   # noqa: E402

COUPLED = (("Gl", -1.0), ("Gku", -1.0), ("Gkl", 1.0), ("Gfu", 1.0), ("Gfl", -1.0))
BAG_BODIES = ("bag", "bag_px", "bag_nx", "bag_py", "bag_ny", "bag_pxpy", "bag_pxny", "bag_nxpy", "bag_nxny")


class Sim:
    dt = 0.002

    def __init__(self, bag=Bag(), board=None, bag_pose=(0.22, 0.0, 0.0), q0=(0, 75, -100, -45, 0), lean=True,
                 servo_latency=0.015, seed=0, record_every=0):
        self.bag = bag
        self.m = mujoco.MjModel.from_xml_string(scene_xml(bag=bag, board=board, lean=lean,
                                                          bag_pos=(bag_pose[0], bag_pose[1], bag.thick / 2 + 0.0005)))
        self.d = mujoco.MjData(self.m)
        m = self.m
        jid = lambda n: m.joint(n).id
        self.qadr = np.array([m.jnt_qposadr[jid(j)] for j in C.JOINTS])
        self.dadr = np.array([m.jnt_dofadr[jid(j)] for j in C.JOINTS])
        self.gq = m.jnt_qposadr[jid("G")]
        self.coupled = [(m.jnt_qposadr[jid(n)], s) for n, s in COUPLED]
        self.act = [m.actuator(j).id for j in C.JOINTS]
        self.gact = m.actuator("G").id
        self.bag_ids = [m.body(b).id for b in BAG_BODIES if bag.model == "floppy" or b == "bag"]
        self.bq = m.jnt_qposadr[jid("bag")]
        # pinch latch (sim/pinch.py)
        self.pinch_eq = [m.equality(f"pinch_{m.body(b).name}").id for b in self.bag_ids]
        self.pad_geoms = {m.geom("pad_fu").id, m.geom("pad_fl").id}
        self.bag_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] in self.bag_ids and m.geom_contype[g]}
        self.bag_mass = float(sum(m.body_mass[b] for b in self.bag_ids))
        self.pinched, self.phi_pinch, self.slipped, self._v_prev = -1, 0.0, False, None
        # bag yaw
        yaw = math.radians(bag_pose[2])
        self.d.qpos[self.bq + 3:self.bq + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        self.set_arm(np.radians(q0), C.phi_for_opening(0.030))
        self.q_tgt = self.q()
        self.g_cmd = self.d.qpos[self.gq]
        self.g_queue = []
        self.lat = servo_latency
        self.t = 0.0
        self.ik = K.Arm(tcp="tip", keep_out=dict(table_z=-1.0, base_r=110.0, base_z=80.0))
        self.frames = [] if record_every else None
        self.record_every = record_every
        self._k = 0
        mujoco.mj_forward(self.m, self.d)
        self.settle(0.3)

    @classmethod
    def from_grasp(cls, grasp, bag=Bag(), board=None, lean=True, record_every=0):
        """Start from a grasp-library state (a real table pickup, bag pinched)."""
        sim = cls(bag, board=board, bag_pose=(0.25, 0.0, 0.0), lean=lean, record_every=record_every)
        sim.d.qpos[:] = grasp["qpos"]
        sim.d.qvel[:] = 0
        k = int(np.ravel(grasp["pinch"])[0])
        sim.m.eq_data[sim.pinch_eq[k]] = grasp["pinch_data"]
        sim.d.eq_active[:] = 0
        sim.d.eq_active[sim.pinch_eq[k]] = 1
        sim.pinched, sim.phi_pinch = k, float(np.ravel(grasp["phi_pinch"])[0])
        sim.q_tgt = np.array(grasp["q_tgt"], float)
        sim.g_cmd, sim.g_target, sim.g_queue = 0.0, 0.0, []
        if sim.frames is not None:
            sim.frames = []
        mujoco.mj_forward(sim.m, sim.d)
        return sim

    # ----------------------------------------------------------- state ----
    def set_arm(self, q, phi):
        self.d.qpos[self.qadr] = q
        self.d.qpos[self.gq] = phi
        for adr, s in self.coupled:
            self.d.qpos[adr] = s * phi
        mujoco.mj_forward(self.m, self.d)

    def q(self):
        return self.d.qpos[self.qadr].copy()

    def tcp(self):
        return (self.d.site("pad_u").xpos + self.d.site("pad_l").xpos) / 2

    def bag_pos(self):
        return self.d.xpos[self.bag_ids[0]].copy()

    def bag_points(self):
        return self.d.xpos[self.bag_ids].copy()

    def bag_yaw(self):
        R = self.d.xmat[self.bag_ids[0]].reshape(3, 3)
        return math.atan2(R[1, 0], R[0, 0])

    # ----------------------------------------------------------- control --
    def step(self, q_t, qd_t, g_target):
        q, qd = self.q(), self.d.qvel[self.dadr]
        ff = self.d.qfrc_bias[self.dadr].copy()              # gravity + Coriolis from the model
        ff[1] -= C.balancer(np, q[1])                         # the spring already helps J2
        tau, _ = C.joint_torque(np, q, qd, q_t, qd_t, tau_ff=ff)
        self.d.ctrl[self.act] = tau
        self.g_queue.append((self.t + self.lat, g_target))
        while len(self.g_queue) > 1 and self.g_queue[1][0] <= self.t:
            self.g_queue.pop(0)
        want = self.g_queue[0][1] if self.g_queue[0][0] <= self.t else self.g_cmd
        rate = math.radians(C.SERVO["speed_dps"]) * self.dt
        self.g_cmd += float(np.clip(want - self.g_cmd, -rate, rate))
        self.d.ctrl[self.gact] = self.g_cmd
        mujoco.mj_step(self.m, self.d)
        self.t += self.dt
        self._k += 1
        if self._k % 5 == 0:
            self._pinch(g_target)
        if self.frames is not None and self._k % self.record_every == 0:
            self.frames.append(self.d.qpos.copy())

    def pads_on_bag(self):
        hit = set()
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if c.dist < 0:
                a, b = c.geom1, c.geom2
                if a in self.pad_geoms and b in self.bag_geoms:
                    hit.add(a)
                elif b in self.pad_geoms and a in self.bag_geoms:
                    hit.add(b)
        return len(hit) == 2

    def _pinch(self, g_target):
        d, m = self.d, self.m
        phi = d.qpos[self.gq]
        g = m.body("grip").id
        if self.pinched < 0:
            if g_target < phi - math.radians(1) and self.pads_on_bag():
                tcp = self.tcp()
                k = int(np.argmin(np.linalg.norm(d.xpos[self.bag_ids] - tcp, axis=1)))
                b = self.bag_ids[k]
                e = self.pinch_eq[k]
                m.eq_data[e] = P.relpose(np, d.xpos[g], d.xmat[g].reshape(3, 3), d.xquat[g], d.xpos[b], d.xquat[b])
                d.eq_active[e] = 1
                self.pinched, self.phi_pinch = k, phi
            self._v_prev = None
            self._p_prev = None
            return
        b = self.bag_ids[self.pinched]
        self.phi_pinch = min(self.phi_pinch, phi)               # tightest squeeze so far
        v = (d.xpos[b] - self._p_prev) / (5 * self.dt) if getattr(self, "_p_prev", None) is not None else np.zeros(3)
        self._p_prev = d.xpos[b].copy()
        release = phi > self.phi_pinch + math.radians(P.RELEASE_DEG)
        if self._v_prev is not None:
            a = (v - self._v_prev) / (5 * self.dt)
            need = self.bag_mass * np.linalg.norm(a - np.array([0, 0, -9.81]))
            if need > P.hold_force(np, phi, C.SERVO["stall_nm"]):
                release, self.slipped = True, True
        self._v_prev = v
        if release:
            d.eq_active[self.pinch_eq[self.pinched]] = 0
            self.pinched = -1
            self._p_prev = None

    def move_joints(self, q_goal, duration, grip=None):
        """Minimum-jerk joint move with velocity feedforward."""
        q0 = self.q_tgt.copy()
        q_goal = np.clip(np.asarray(q_goal, float), C.QLO, C.QHI)
        g = self.g_target if grip is None else grip
        n = max(1, int(duration / self.dt))
        for i in range(n):
            s = (i + 1) / n
            f = s ** 3 * (10 - 15 * s + 6 * s * s)
            fd = (30 * s * s - 60 * s ** 3 + 30 * s ** 4) / duration
            self.q_tgt = q0 + (q_goal - q0) * f
            self.step(self.q_tgt, (q_goal - q0) * fd, g)
        self.g_target = g

    def settle(self, seconds, grip=None):
        g = getattr(self, "g_target", self.d.qpos[self.gq]) if grip is None else grip
        self.g_target = g
        for _ in range(int(seconds / self.dt)):
            self.step(self.q_tgt, np.zeros(5), g)

    def solve(self, p, pitch, roll, grip_mm=20.0):
        """IK for the pad tips at p (m) with the given approach; returns joint angles (rad) or None."""
        seed = dict(zip(C.JOINTS, np.degrees(self.q_tgt)))
        sols = []
        for pt in (pitch, pitch + 5, pitch + 10, pitch + 15, pitch + 20, pitch + 25, pitch + 30, pitch + 35):
            try:
                s = self.ik.ik(p[0] * 1000, p[1] * 1000, p[2] * 1000, pitch=pt, roll=roll, grip=grip_mm, seed=seed)
            except K.Unreachable:
                continue
            if s.ok:
                return np.radians([s.q[j] for j in C.JOINTS])
            sols.append(s)
        return None

    def move_tip(self, p, pitch, roll, duration, grip=None, grip_mm=20.0):
        q = self.solve(p, pitch, roll, grip_mm)
        if q is None:
            return False
        self.move_joints(q, duration, grip)
        return True


def pick(sim: Sim, strategy="ridge", press=0.006, lift=0.10, pose_noise=0.0, rng=None, approach_pitch=None,
         tilt=20.0, edge_out=0.008):
    """Pick the floppy bag off the table.

    strategy "ridge" (default): gripper pointing down, jaws across the bag; the
        pads press on it and pinch it as they close (the pinch latch in
        sim/pinch.py holds the cloth).
             "edge": tilt the gripper `tilt` deg off vertical so one pad
        hangs ~13 mm lower than the other.  Take the bag edge that faces the arm:
        the low pad comes down just outside that edge, the high pad lands on the
        edge bead, and closing traps the edge between them, like a thumb on
        top and a finger under it.
    pose_noise: camera error on the bag position (m) and yaw (x 10 deg / cm).
    Returns (success, info)."""
    rng = rng or np.random.default_rng()
    b = sim.bag
    p = sim.bag_pos()
    yaw = sim.bag_yaw()
    if pose_noise:
        p[:2] += rng.normal(0, pose_noise, 2)
        yaw += rng.normal(0, math.radians(1000 * pose_noise))
    open_phi, closed = C.G_MAX, 0.0
    if strategy == "edge":
        # the edge whose outward normal points most toward the arm's base
        rhat = p[:2] / np.linalg.norm(p[:2])
        normals = [np.array([math.cos(yaw + k * math.pi / 2), math.sin(yaw + k * math.pi / 2)]) for k in range(4)]
        n = min(normals, key=lambda v: v @ rhat)
        center = p.copy()
        center[:2] += n * (b.size / 2 + edge_out)
        ax = math.atan2(-n[1], -n[0])                    # jaw axis: from the low (outside) pad toward the bag
        pitch = -90.0 + tilt if approach_pitch is None else approach_pitch
        tip_z = 0.5 * (0.003 + b.thick) - press + 0.004
    else:
        ax = yaw
        center = p.copy()
        pitch = -90.0 if approach_pitch is None else approach_pitch
        tip_z = b.thick - press
    theta = math.atan2(center[1], center[0])
    roll = math.degrees(theta - ax)
    above = center + np.array([0, 0, b.thick + 0.06])
    if not sim.move_tip(above, pitch, roll, 1.0, grip=open_phi, grip_mm=37):
        return False, dict(reason="unreachable")
    sim.settle(0.15)
    if not sim.move_tip(np.r_[center[:2], tip_z], pitch, roll, 0.6, grip=open_phi, grip_mm=37):
        return False, dict(reason="unreachable")
    sim.settle(0.15)
    sim.settle(0.5, grip=closed)                     # close: the servo squeezes up to its stall torque
    if not sim.move_tip(np.r_[center[:2], tip_z + lift], pitch, roll, 0.8, grip=closed, grip_mm=10):
        return False, dict(reason="unreachable")
    sim.settle(0.5)
    pts = sim.bag_points()
    held = pts[:, 2].min() > 0.02 and np.linalg.norm(sim.bag_pos() - sim.tcp()) < 0.06
    return bool(held), dict(reason="ok" if held else "dropped", bag_low=float(pts[:, 2].min()))


def throw(sim: Sim, params, follow=1.6, stop_at_landing=False, on_step=None):
    """Open-loop overhand throw (the throw study's 13 parameters): wind up from
    wherever the arm is, swing J2-J4 along minimum-jerk profiles, open the
    gripper at the release time.  Returns what happened: when the bag left the
    pinch (by the jaws opening or by slipping), its speed and angle then, and
    where it first touched the floor."""
    import throw_study as T
    p = np.clip(np.asarray(params, float), T.LO, T.HI)
    q0, q1, ts, du, t_rel = p[0:3], p[3:6], p[6:9], p[9:12], p[12]
    qa = np.radians([0.0, *q0, 0.0])
    qb = np.radians([0.0, *q1, 0.0])
    out = dict(released_at=None, slipped=False, landed_xy=None, held=True)
    sim.move_joints(qa, T.SETTLE * 0.8, grip=0.0)
    sim.settle(T.SETTLE * 0.2, grip=0.0)
    if sim.pinched < 0:
        out["held"] = False
        return out
    n = int((T.T_THROW + follow) / sim.dt)
    b0 = sim.bag_ids[0]
    for k in range(n):
        t = k * sim.dt
        if t < T.T_THROW:
            q_t, qd_t = qa.copy(), np.zeros(5)
            for i in range(3):
                f, fd = T.minjerk((t - ts[i]) / du[i])
                q_t[i + 1] = qa[i + 1] + (qb[i + 1] - qa[i + 1]) * f
                qd_t[i + 1] = (qb[i + 1] - qa[i + 1]) * fd / du[i] if 0 < (t - ts[i]) / du[i] < 1 else 0.0
            sim.q_tgt = np.clip(q_t, C.QLO, C.QHI)
        else:
            qd_t = np.zeros(5)
        sim.step(sim.q_tgt, qd_t, C.G_MAX if t >= t_rel else 0.0)
        if on_step is not None:
            on_step(sim)
        if out["released_at"] is None and sim.pinched < 0:
            v = sim.d.qvel[sim.m.jnt_dofadr[sim.m.joint("bag").id]:][:3].copy()
            out.update(released_at=t, slipped=bool(sim.slipped) and t < t_rel + 0.03,
                       release_speed=float(np.linalg.norm(v)), release_angle=float(math.degrees(math.atan2(v[2], math.hypot(v[0], v[1])))),
                       release_height=float(sim.d.xpos[b0][2]))
        if out["released_at"] is not None and out["landed_xy"] is None:
            bp = sim.d.xpos[b0]
            if bp[2] < sim.bag.size / 2 + 0.002 and math.hypot(bp[0], bp[1]) > 0.13:
                out["landed_xy"] = bp[:2].tolist()
                if stop_at_landing:
                    break
    sim.g_target = C.G_MAX
    if out["landed_xy"] is None:
        out["landed_xy"] = sim.d.xpos[b0][:2].tolist()
    return out
