"""Actuator model shared by the GPU envs (jax.numpy) and the CPU throw study (numpy).

Each joint is a moteus-c1 running a position/velocity PD loop on a CubeMars
gimbal motor that drives the joint through a direct capstan:

    tau_joint_cmd = kp (q* - q) + kd (qd* - qd)            (moteus PD, joint units)
    tau_motor     = tau_joint_cmd / (ratio * eff)
    tau_motor     clipped to the motor's torque-speed envelope at 24 V:
                      upper = min(peak,  kt (V - ke w) / R)
                      lower = max(-peak, -kt (V + ke w) / R)     (w = motor speed)
    tau_joint     = tau_motor * ratio * eff  (+ shoulder balancer spring on J2)

The torque-speed envelope is what limits how fast the arm can swing, so it is
what limits a throw.  Data from the CubeMars datasheets (sim/assets/arm_meta.json).
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
META = json.load(open(os.path.join(HERE, "assets", "arm_meta.json")))
JOINTS = ("J1", "J2", "J3", "J4", "J5")

_J = META["joints"]
RATIO = np.array([_J[j]["ratio"] for j in JOINTS])
KT = np.array([_J[j]["kt"] for j in JOINTS])
RES = np.array([_J[j]["R"] for j in JOINTS])
TPK = np.array([_J[j]["tau_peak"] for j in JOINTS])
TRATED = np.array([_J[j]["tau_rated"] for j in JOINTS])
KE = np.array([60.0 / (2 * math.pi * _J[j]["kv_rpm"]) for j in JOINTS])
QLO = np.array([_J[j]["rom"][0] for j in JOINTS])
QHI = np.array([_J[j]["rom"][1] for j in JOINTS])
EFF = META["capstan_eff"]
VBUS = META["v_bus"]
BAL_Q = np.radians(np.array(META["balancer"]["q2_deg"], float))
BAL_T = np.array(META["balancer"]["torque_nm"], float)

# PD gains at the joint (N m / rad, N m s / rad).  Roughly 12-15 Hz bandwidth
# on each joint's reflected inertia; the policy can scale them per episode.
KP = np.array([60.0, 160.0, 90.0, 22.0, 8.0])
KD = np.array([2.0, 5.0, 3.0, 0.7, 0.25])

# joint-space speed limits (rad/s): the no-load speed of each motor at 24 V
QD_MAX = (VBUS / KE) / RATIO

SERVO = META["servo"]
G_MAX = math.radians(META["grip"]["phi_max"])


def balancer(xp, q2):
    """Shoulder balancer spring torque at J2 (N m)."""
    return xp.interp(q2, BAL_Q, BAL_T)


def joint_torque(xp, q, qd, q_tgt, qd_tgt, kp_scale=1.0, torque_scale=1.0, v_bus=VBUS, tau_ff=0.0):
    """Joint torques (N m) for J1..J5, and the motor currents (A) behind them.
    tau_ff: feedforward joint torque (e.g. gravity compensation from the
    model, which moteus accepts as feedforward_torque alongside position)."""
    tau_cmd = KP * kp_scale * (q_tgt - q) + KD * kp_scale * (qd_tgt - qd) + tau_ff
    w = qd * RATIO                                             # motor speed, rad/s
    tm = tau_cmd / (RATIO * EFF)
    peak = TPK * torque_scale
    upper = xp.minimum(peak, KT * (v_bus - KE * w) / RES)
    lower = xp.maximum(-peak, -KT * (v_bus + KE * w) / RES)
    tm = xp.clip(tm, lower, upper)
    current = tm / KT
    tau = tm * RATIO * EFF
    # shoulder balancer (passive spring, always acting)
    tau = tau + xp.array([0.0, 1.0, 0.0, 0.0, 0.0]) * balancer(xp, q[1])
    return tau, current


def heat_w(xp, current):
    """Copper loss per motor (W), for penalties and thermal checks."""
    return current * current * RES


def grip_opening(phi):
    """Pad gap (m) for the gripper link angle phi (rad)."""
    g = META["grip"]
    return 2 * (g["PZ"] + g["LL"] * math.sin(phi) - g["FACE"]) / 1000.0


def phi_for_opening(gap_m):
    g = META["grip"]
    s = (gap_m * 1000.0 / 2 + g["FACE"] - g["PZ"]) / g["LL"]
    return math.asin(max(-1.0, min(1.0, s)))
