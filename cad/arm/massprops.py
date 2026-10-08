"""Rough mass properties per body (for balancer sizing and torque estimates)."""
import numpy as np

PETG = 1.27e-3            # g/mm^3
FILL = 0.55               # 4 walls + 30 % gyroid on parts this size
STEEL = 7.85e-3
REAL_MASS = {             # purchased parts where the model volume is not the mass
    "servo_body": 13.4,
    "enc_board": 2.5,
    "magnet": 0.5,
    "magnet_drv": 0.8,
    "drv_pcb": 15.0,      # moteus-c1
}
ZERO = ("motor_bell", "motor_stator", "motor_shaft", "motor_wires", "cable", "servo_spline", "drv_parts", "drv_chip")


def part_mass(p):
    if p.key == "motor_base":                 # whole motor mass, from the spec sheet
        from .params import MOTORS
        for m in MOTORS.values():
            if m["name"] == p.bom:
                return float(m["mass_g"])
    if p.key in REAL_MASS:
        return REAL_MASS[p.key]
    if p.key.startswith(ZERO):
        return 0.0
    v = p.local_shape().Volume()
    if p.kind == "printed":
        dens = PETG * (1.0 if v < 2500 else FILL)
        if p.key.startswith(("finray", "act_spider")):
            dens = 1.21e-3 * 0.6
        return v * dens
    if p.key.startswith("brg_"):
        return v * STEEL * 0.75
    return v * STEEL


_cache = {}


def body_props(bodies):
    """{body: (mass_g, com_xyz in body frame)}"""
    out = {}
    for name, parts in bodies.items():
        m_tot, mc = 0.0, np.zeros(3)
        for p in parts:
            key = (id(p.shape), p.key)
            if key not in _cache:
                s = p.local_shape()
                _cache[key] = (part_mass(p), np.array(s.Center().toTuple()) if part_mass(p) > 0 else np.zeros(3))
            m, c = _cache[key]
            if m <= 0:
                continue
            cw = (p.M @ np.r_[c, 1.0])[:3]
            m_tot += m
            mc += m * cw
        out[name] = (m_tot, mc / max(m_tot, 1e-9))
    return out


def gravity_torque_j2(props, frames, moving=("upper", "fore", "hand", "grip", "l1u", "l1l", "l2u", "l2l", "fu", "fl"),
                      payload_g=0.0, tool=None):
    """Torque (N m) gravity applies about J2, positive = lifting direction (CCW in link plane)."""
    Fu = frames["upper"]
    o = Fu[:3, 3]
    axis = Fu[:3, :3] @ np.array([0, -1.0, 0])     # +J2 rotation axis in world
    tau = 0.0
    g = np.array([0, 0, -9.81])
    for b in moving:
        m, c = props[b]
        cw = (frames[b] @ np.r_[c, 1.0])[:3]
        f = g * m / 1000.0
        tau += np.dot(np.cross((cw - o) / 1000.0, f), axis)
    if payload_g and tool is not None:
        tau += np.dot(np.cross((tool - o) / 1000.0, g * payload_g / 1000.0), axis)
    return tau


DOWNSTREAM = {
    "J2": ("upper", "fore", "hand", "grip", "l1u", "l1l", "l2u", "l2l", "fu", "fl"),
    "J3": ("fore", "hand", "grip", "l1u", "l1l", "l2u", "l2l", "fu", "fl"),
    "J4": ("hand", "grip", "l1u", "l1l", "l2u", "l2l", "fu", "fl"),
}
JOINT_FRAME = {"J2": "upper", "J3": "fore", "J4": "hand"}


def gravity_torque(props, frames, joint, payload_g=0.0, tool=None):
    """Gravity torque (N m) about a pitch joint, + = lifting direction."""
    F0 = frames[JOINT_FRAME[joint]]
    o = F0[:3, 3]
    axis = F0[:3, :3] @ np.array([0, -1.0, 0])
    g = np.array([0, 0, -9.81])
    tau = 0.0
    for b in DOWNSTREAM[joint]:
        m, c = props[b]
        cw = (frames[b] @ np.r_[c, 1.0])[:3]
        tau += np.dot(np.cross((cw - o) / 1000.0, g * m / 1000.0), axis)
    if payload_g and tool is not None:
        tau += np.dot(np.cross((tool - o) / 1000.0, g * payload_g / 1000.0), axis)
    return tau
