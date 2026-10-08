"""Capstan Arm 5: forward and inverse kinematics, plus motor commands.

Standalone (numpy only), so it can run on whatever drives the arm.

You give the gripper a target with six numbers:

    x, y, z   tool point in mm, base frame (origin at the centre of the base's
              underside, z up, x forward at J1 = 0)
    pitch     tool approach angle in degrees above horizontal, measured in the
              vertical plane through the base axis and the target.
              0 = pointing straight out, -90 = pointing straight down,
              +90 = straight up; beyond -90 it tilts back toward the base.
    roll      jaw angle about the approach axis, degrees.  0 = the jaws open
              and close in that vertical plane (one finger above the other);
              90 = they pinch side to side.  The jaws are symmetric, so roll
              and roll +/- 180 grip the same way; the solver uses whichever
              keeps J5 in range.
    grip      gap between the pads in mm (1 to 38)

and get back the five joint angles, the moteus position for each motor and
the gripper servo angle.

The arm has five joints, so it can't reach every orientation: the tool
always points within the vertical plane through the base axis (that's where
"pitch" lives).  Yaw comes from J1.

Example
-------
    from capstan_ik import Arm
    arm = Arm()
    sol = arm.ik(250, 60, 120, pitch=-60, roll=0, grip=20)
    print(sol)                 # joint angles, motor revs, warnings
    sol.q                      # {'J1': .., 'J2': .., ..., 'G': ..}
    sol.motors                 # {'J1': rev, ..., 'servo': deg}
    arm.fk(sol.q)              # back to x, y, z, pitch, roll, grip

    python capstan_ik.py 250 60 120 --pitch -60 --roll 0 --grip 20
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# ------------------------------------------------------------- geometry -----
# All from the CAD (arm/params.py, arm/arm.py, arm/hand.py).
# ik/test_ik.py checks every number here against the CAD model.
GEOM = dict(
    H_SHOULDER=155.0,     # J2 axis height above the base underside (J2_Z)
    L1=150.0,             # shoulder -> elbow
    L2=130.0,             # elbow -> wrist pitch
    XG=118.5,             # wrist pitch axis -> roll flange front face (grip frame)
    XP=34.0,              # grip frame -> gripper link pivots
    PZ=6.0,               # link pivot offset from the gripper centre line
    LL=24.0,              # parallel-link length
    FACE=5.5,             # pad face inset from the finger frame
    PAD_X=(8.0, 52.0),    # Fin Ray extent along the finger (finger frame)
)
TCP_MODES = {"center": 30.0, "tip": 52.0}   # where on the pad the tool point sits

LIMITS = {   # deg, collision-checked ranges from params.py
    "J1": (-150.0, 150.0), "J2": (-10.0, 120.0), "J3": (-110.0, 110.0),
    "J4": (-90.0, 70.0), "J5": (-150.0, 150.0), "G": (0.0, 50.0),
}

# Motors.  moteus position is in rotor revolutions.
RATIO = {"J1": 9.7, "J2": 10.4, "J3": 8.0, "J4": 6.5, "J5": 6.0}   # capstan Rc / pulley r
# An open capstan turns the pulley opposite to the drum, hence -1.  Which way a
# motor counts depends on how it's mounted and wired, so check each joint on
# its first slow jog and flip its sign here if it moves the wrong way.
MOTOR_SIGN = {"J1": -1, "J2": -1, "J3": -1, "J4": -1, "J5": -1}
# Rotor position (rev) when the joint is at 0 deg.  Set after homing: put the
# joint at its zero, read the moteus position, store it here.
MOTOR_OFFSET = {"J1": 0.0, "J2": 0.0, "J3": 0.0, "J4": 0.0, "J5": 0.0}
CAN_ID = {"J1": 1, "J2": 2, "J3": 3, "J4": 4, "J5": 5}
# Gripper servo: centred (90 deg) with the jaws closed, then +1 deg per deg of
# link angle.  Flip SERVO_SIGN if it opens the wrong way.
SERVO_CENTER, SERVO_SIGN = 90.0, 1.0

# Approximate keep-out volumes (mm), checked at the elbow, wrist, tool point
# and fingertips.  The joint limits are collision-checked on their own, but some
# combinations can still reach into the base or the table.
KEEP_OUT = dict(
    table_z=8.0,                  # stay this far above the table
    base_r=115.0, base_z=90.0,    # base puck (r 100, h 72) plus margin
    turret_r=75.0, turret_z=200.0,  # turret, shoulder actuator and balancer around the J1 axis
)

JOINTS = ("J1", "J2", "J3", "J4", "J5")


# --------------------------------------------------------------- helpers ----
def _rz(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    M = np.eye(4)
    M[:2, :2] = [[c, -s], [s, c]]
    return M


def _ry_link(deg):
    """Joint rotation used by the CAD: about -y, so positive lifts +x toward +z."""
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    M = np.eye(4)
    M[0, 0], M[0, 2], M[2, 0], M[2, 2] = c, -s, s, c
    return M


def _rx(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    M = np.eye(4)
    M[1:3, 1:3] = [[c, -s], [s, c]]
    return M


def _t(x=0.0, y=0.0, z=0.0):
    M = np.eye(4)
    M[:3, 3] = (x, y, z)
    return M


def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def grip_to_phi(gap_mm):
    """Pad gap (mm) -> gripper link angle (deg)."""
    g = GEOM
    s = (gap_mm / 2 + g["FACE"] - g["PZ"]) / g["LL"]
    return math.degrees(math.asin(min(1.0, max(-1.0, s))))


def phi_to_grip(phi):
    g = GEOM
    return 2 * (g["PZ"] + g["LL"] * math.sin(math.radians(phi)) - g["FACE"])


def tool_length(phi, tcp="center"):
    """Wrist pitch axis -> tool point, along the hand axis (mm)."""
    g = GEOM
    return g["XG"] + g["XP"] + g["LL"] * math.cos(math.radians(phi)) + TCP_MODES[tcp]


# ---------------------------------------------------------------- types -----
@dataclass
class Pose:
    x: float
    y: float
    z: float
    pitch: float
    roll: float
    grip: float
    T: np.ndarray = field(repr=False, default=None)     # 4x4 tool frame: x = approach, z = jaw axis

    def __str__(self):
        return (f"x {self.x:8.2f}  y {self.y:8.2f}  z {self.z:8.2f} mm   pitch {self.pitch:7.2f}  "
                f"roll {self.roll:7.2f} deg   grip {self.grip:5.1f} mm")


@dataclass
class Solution:
    q: dict                      # joint angles, deg (J1..J5) and gripper link angle G
    ok: bool                     # within limits and keep-outs
    problems: list
    motors: dict                 # moteus position (rev) per joint, servo angle (deg)
    branch: str                  # "front/elbow-up" etc.
    cost: float = 0.0            # distance from the seed (deg), used for ranking

    def __str__(self):
        j = "  ".join(f"{k} {self.q[k]:8.2f}" for k in (*JOINTS, "G"))
        m = "  ".join(f"{k} {v:+8.4f}" for k, v in self.motors.items() if k != "servo")
        s = [f"{'OK' if self.ok else 'NOT OK'}  [{self.branch}]",
             f"  joints (deg): {j}",
             f"  motors (rev): {m}   servo {self.motors['servo']:.1f} deg"]
        s += [f"  ! {p}" for p in self.problems]
        return "\n".join(s)


class Unreachable(ValueError):
    pass


# ------------------------------------------------------------------ arm -----
class Arm:
    def __init__(self, tcp="center", limits=None, keep_out=None):
        """tcp: 'center' (middle of the Fin Ray pads) or 'tip' (pad tips)."""
        self.tcp = tcp
        self.limits = dict(LIMITS, **(limits or {}))
        self.keep_out = dict(KEEP_OUT, **(keep_out or {}))

    # ------------------------------------------------------------ FK -------
    def frames(self, q):
        """World 4x4 frames: shoulder, elbow, wrist (pitch axis), grip (after roll), tool."""
        Ft = _rz(q["J1"])
        Fu = Ft @ _t(0, 0, GEOM["H_SHOULDER"]) @ _ry_link(q["J2"])
        Ff = Fu @ _t(GEOM["L1"]) @ _ry_link(q["J3"])
        Fh = Ff @ _t(GEOM["L2"]) @ _ry_link(q["J4"])
        Fg = Fh @ _t(GEOM["XG"]) @ _rx(q["J5"])
        phi = q.get("G", 0.0)
        Ftool = Fg @ _t(tool_length(phi, self.tcp) - GEOM["XG"])
        tips = [Fg @ _t(GEOM["XP"] + GEOM["LL"] * math.cos(math.radians(phi)) + GEOM["PAD_X"][1], 0,
                        s * (GEOM["PZ"] + GEOM["LL"] * math.sin(math.radians(phi)))) for s in (1, -1)]
        return dict(turret=Ft, shoulder=Fu, elbow=Ff, wrist=Fh, grip=Fg, tool=Ftool,
                    tip_a=tips[0], tip_b=tips[1])

    def fk(self, q):
        """Joint angles (deg) -> Pose."""
        F = self.frames(q)
        T = F["tool"]
        p, a, j = T[:3, 3], T[:3, 0], T[:3, 2]
        rho = math.hypot(p[0], p[1])
        theta = math.atan2(p[1], p[0]) if rho > 1e-6 else math.radians(q["J1"])
        rhat = np.array([math.cos(theta), math.sin(theta), 0.0])
        pitch = math.degrees(math.atan2(a[2], a @ rhat))
        u, w = self._roll_ref(theta, pitch)
        roll = math.degrees(math.atan2(j @ w, j @ u))
        return Pose(*p, pitch, roll, phi_to_grip(q.get("G", 0.0)), T)

    @staticmethod
    def _roll_ref(theta, pitch):
        """Roll reference: u = 'up' in the vertical plane, perpendicular to the approach."""
        p = math.radians(pitch)
        rhat = np.array([math.cos(theta), math.sin(theta), 0.0])
        a = math.cos(p) * rhat + np.array([0, 0, math.sin(p)])
        u = -math.sin(p) * rhat + np.array([0, 0, math.cos(p)])
        return u, np.cross(a, u)

    # ------------------------------------------------------------ IK -------
    def ik_all(self, x, y, z, pitch=-90.0, roll=0.0, grip=20.0, seed=None):
        """Every solution (up to 4: front/back x elbow up/down), best first.
        Each one is checked against the joint limits and keep-outs."""
        seed = seed or {}
        phi = grip_to_phi(grip)
        d = tool_length(phi, self.tcp)
        rho = math.hypot(x, y)
        theta = math.atan2(y, x) if rho > 1e-6 else math.radians(seed.get("J1", 0.0))
        u, w = self._roll_ref(theta, pitch)
        jaw = math.cos(math.radians(roll)) * u + math.sin(math.radians(roll)) * w
        sols, reach = [], []
        for branch, j1, r_loc, phi_loc in (("front", math.degrees(theta), rho, pitch),
                                           ("back", math.degrees(theta) + 180.0, -rho, 180.0 - pitch)):
            # wrist pitch axis position in the arm plane (r along the arm's local x, up)
            wr = r_loc - d * math.cos(math.radians(phi_loc))
            wz = z - GEOM["H_SHOULDER"] - d * math.sin(math.radians(phi_loc))
            L1, L2 = GEOM["L1"], GEOM["L2"]
            D2 = wr * wr + wz * wz
            c3 = (D2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
            if abs(c3) > 1.0:
                reach.append(math.sqrt(D2) - (L1 + L2) if c3 > 1 else abs(L1 - L2) - math.sqrt(D2))
                continue
            for elbow, sgn in (("elbow-up", -1.0), ("elbow-down", 1.0)):
                j3 = sgn * math.degrees(math.acos(c3))
                j2 = math.degrees(math.atan2(wz, wr) - math.atan2(L2 * math.sin(math.radians(j3)),
                                                                   L1 + L2 * math.cos(math.radians(j3))))
                j4 = _wrap(phi_loc - j2 - j3)
                q = {"J1": _wrap(j1), "J2": _wrap(j2), "J3": j3, "J4": j4, "J5": 0.0, "G": phi}
                # roll: J5 that puts the jaw axis where it was asked for (or its mirror)
                Fh = self.frames(q)["wrist"]
                q5 = math.degrees(math.atan2(-(jaw @ Fh[:3, 1]), jaw @ Fh[:3, 2]))
                cands = [_wrap(q5), _wrap(q5 + 180.0)]
                lo, hi = self.limits["J5"]
                inr = [c for c in cands if lo <= c <= hi] or cands
                q["J5"] = min(inr, key=lambda c: abs(c - seed.get("J5", 0.0)))
                sols.append(self._finish(q, f"{branch}/{elbow}", seed))
        if not sols:
            short = min(reach) if reach else float("nan")
            raise Unreachable(f"target out of reach by {short:.1f} mm (wrist must be within "
                              f"{abs(GEOM['L1'] - GEOM['L2']):.0f}-{GEOM['L1'] + GEOM['L2']:.0f} mm "
                              f"of the shoulder)")
        # best first: valid before invalid, then front/elbow-up preference, then closest to the seed
        pref = {"front/elbow-up": 0, "front/elbow-down": 1, "back/elbow-up": 2, "back/elbow-down": 3}
        sols.sort(key=lambda s: (not s.ok, s.cost if seed else 0.0, pref[s.branch]))
        return sols

    def ik(self, x, y, z, pitch=-90.0, roll=0.0, grip=20.0, seed=None, strict=False):
        """Best solution.  seed: current joint angles, so moves stay on the same
        branch and J1/J5 don't take the long way round.  strict=True raises
        if no solution is within limits."""
        sols = self.ik_all(x, y, z, pitch, roll, grip, seed)
        if strict and not sols[0].ok:
            raise Unreachable("no solution inside the limits:\n" + "\n".join(str(s) for s in sols))
        return sols[0]

    def ik_pose(self, T, grip=20.0, seed=None):
        """Best effort for a full 4x4 tool pose (x = approach, z = jaw axis).
        The approach is projected into the reachable vertical plane; returns
        (solution, orientation error in deg)."""
        p, a, j = T[:3, 3], T[:3, 0], T[:3, 2]
        theta = math.atan2(p[1], p[0])
        rhat = np.array([math.cos(theta), math.sin(theta), 0.0])
        pitch = math.degrees(math.atan2(a[2], a @ rhat))
        u, w = self._roll_ref(theta, pitch)
        roll = math.degrees(math.atan2(j @ w, j @ u))
        sol = self.ik(*p, pitch, roll, grip, seed)
        Ta = self.fk(sol.q).T
        err = math.degrees(math.acos(max(-1.0, min(1.0, Ta[:3, 0] @ a))))
        return sol, err

    # ------------------------------------------------------- checks --------
    def _finish(self, q, branch, seed):
        problems = []
        for k, (lo, hi) in self.limits.items():
            if not lo - 1e-6 <= q[k] <= hi + 1e-6:
                problems.append(f"{k} = {q[k]:.1f} deg is outside {lo:.0f}..{hi:.0f}")
        problems += self.check_keep_out(q)
        cost = sum(abs(_wrap(q[k] - seed[k])) for k in JOINTS if k in seed) if seed else 0.0
        return Solution(q=q, ok=not problems, problems=problems, motors=self.to_motors(q),
                        branch=branch, cost=cost)

    def check_keep_out(self, q):
        K = self.keep_out
        F = self.frames(q)
        out = []
        for name in ("elbow", "wrist", "tool", "tip_a", "tip_b"):
            x, y, z = F[name][:3, 3]
            r = math.hypot(x, y)
            if z < K["table_z"]:
                out.append(f"{name} is {z:.0f} mm above the table (min {K['table_z']:.0f})")
            elif r < K["base_r"] and z < K["base_z"]:
                out.append(f"{name} is inside the base keep-out (r {r:.0f}, z {z:.0f})")
            elif name != "elbow" and r < K["turret_r"] and z < K["turret_z"]:
                out.append(f"{name} is inside the turret keep-out (r {r:.0f}, z {z:.0f})")
        return out

    # ------------------------------------------------------- motors --------
    @staticmethod
    def to_motors(q):
        """Joint angles (deg) -> moteus positions (rotor rev) and servo angle (deg)."""
        m = {k: MOTOR_OFFSET[k] + MOTOR_SIGN[k] * RATIO[k] * q[k] / 360.0 for k in JOINTS}
        m["servo"] = SERVO_CENTER + SERVO_SIGN * q.get("G", 0.0)
        return m

    @staticmethod
    def from_motors(m):
        """moteus positions (rev) -> joint angles (deg).  Use the joint encoders
        instead when you have them; this is the motor-side estimate."""
        q = {k: (m[k] - MOTOR_OFFSET[k]) * 360.0 / (MOTOR_SIGN[k] * RATIO[k]) for k in JOINTS}
        if "servo" in m:
            q["G"] = (m["servo"] - SERVO_CENTER) / SERVO_SIGN
        return q

    # --------------------------------------------------- differential ------
    def jacobian(self, q, eps=1e-4):
        """6x5: d[x, y, z (mm), rx, ry, rz (rad)] / d[J1..J5 (rad)], world frame."""
        T0 = self.frames(q)["tool"]
        J = np.zeros((6, 5))
        for i, k in enumerate(JOINTS):
            qq = dict(q)
            qq[k] = q[k] + math.degrees(eps)
            T1 = self.frames(qq)["tool"]
            J[:3, i] = (T1[:3, 3] - T0[:3, 3]) / eps
            dR = T1[:3, :3] @ T0[:3, :3].T
            J[3:, i] = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / (2 * eps)
        return J

    def joint_speeds(self, q, v_mm_s=(0, 0, 0), w_deg_s=(0, 0, 0)):
        """Joint speeds (deg/s) for a tool velocity (damped least squares, so
        it stays sane near singularities) and the matching motor speeds (rev/s)."""
        J = self.jacobian(q)
        v = np.r_[np.asarray(v_mm_s, float), np.radians(np.asarray(w_deg_s, float))]
        W = np.diag([1, 1, 1, 200, 200, 200])          # weigh 1 rad as 200 mm
        lam = 1e-3
        A = J.T @ W @ J + lam * np.eye(5)
        qd = np.degrees(np.linalg.solve(A, J.T @ W @ v))
        qd = dict(zip(JOINTS, qd))
        md = {k: MOTOR_SIGN[k] * RATIO[k] * qd[k] / 360.0 for k in JOINTS}
        return qd, md

    def manipulability(self, q):
        """Positional manipulability (bigger = farther from a singularity)."""
        J = self.jacobian(q)[:3]
        return float(math.sqrt(max(0.0, np.linalg.det(J @ J.T))))

    # --------------------------------------------------------- paths -------
    def line(self, start, end, steps=50, seed=None):
        """Straight-line move.  start/end: dicts with x, y, z, pitch, roll, grip.
        Returns the list of solutions; raises if any step is unreachable,
        out of limits or flips branch."""
        out, s = [], seed
        branch = None
        for i in range(steps + 1):
            t = i / steps
            t = t * t * (3 - 2 * t)                          # ease in / out
            tgt = {k: start[k] + (end[k] - start[k]) * t for k in start}
            sol = self.ik(**tgt, seed=s)
            if not sol.ok:
                raise Unreachable(f"step {i}/{steps} at {tgt}:\n{sol}")
            if branch and sol.branch != branch:
                raise Unreachable(f"step {i}/{steps}: path changes from {branch} to {sol.branch}")
            branch, s = sol.branch, sol.q
            out.append(sol)
        return out


# -------------------------------------------------------------------- CLI ---
def _main():
    import argparse
    ap = argparse.ArgumentParser(description="Capstan Arm 5 inverse kinematics")
    ap.add_argument("x", type=float)
    ap.add_argument("y", type=float)
    ap.add_argument("z", type=float)
    ap.add_argument("--pitch", type=float, default=-90.0, help="deg above horizontal (default -90: straight down)")
    ap.add_argument("--roll", type=float, default=0.0, help="deg about the approach axis")
    ap.add_argument("--grip", type=float, default=20.0, help="pad gap, mm")
    ap.add_argument("--tcp", choices=list(TCP_MODES), default="center")
    ap.add_argument("--all", action="store_true", help="show every branch")
    a = ap.parse_args()
    arm = Arm(tcp=a.tcp)
    try:
        sols = arm.ik_all(a.x, a.y, a.z, a.pitch, a.roll, a.grip)
    except Unreachable as e:
        raise SystemExit(str(e))
    for s in (sols if a.all else sols[:1]):
        print(s)
        print("  check:", arm.fk(s.q))


if __name__ == "__main__":
    _main()
