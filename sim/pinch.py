"""Pinch latch: how the Fin Ray pads hold a floppy beanbag.

A real pinch squeezes the beans aside and clamps folded cloth between the pads.
Nine rigid beads can't fold like cloth, so the pinch is modelled as a latch:

  grab     while the gripper is closing, as soon as both pads are pressing on the
           bag, the bead nearest the pinch point is welded to the gripper (a
           slightly springy weld, like cloth) where it is.  The rest of the bag
           still hangs and flops on its hinges.
  release  when the jaws have actually opened (servo delay and speed included)
           by more than RELEASE_DEG past where they pinched.
  slip     when holding the bag would take more force than the pinch can give:
           F_hold = 2 x friction x pad force, pad force = servo torque / (2 L cos phi)
           (the servo drives both jaws through the gears).  About 9 N for an
           MG90S: plenty for a 25 g bag, about 2 g for a 450 g one.

Works with numpy (CPU MuJoCo, sim/skills.py) and jax.numpy (MJX envs).
"""
import math

import numpy as np

RELEASE_DEG = 3.0
PAD_FRICTION = 1.0          # TPU on bag fabric
L_LINK = 0.024              # m, gripper parallel link
CONTACT_FLOOR = 0.0


def hold_force(xp, phi, stall_nm):
    """Most force (N) the pinch can resist before the bag slides out."""
    pad_force = stall_nm / (2 * L_LINK * xp.cos(phi))
    return 2 * PAD_FRICTION * pad_force


def quat_mul(xp, a, b):
    w1, x1, y1, z1 = a[0], a[1], a[2], a[3]
    w2, x2, y2, z2 = b[0], b[1], b[2], b[3]
    return xp.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def relpose(xp, p1, R1, q1, p2, q2):
    """Weld eq_data (11 numbers) that holds body2 where it is relative to body1."""
    rel_p = R1.T @ (p2 - p1)
    q1c = q1 * xp.array([1.0, -1.0, -1.0, -1.0])
    rel_q = quat_mul(xp, q1c, q2)
    return xp.concatenate([xp.zeros(3), rel_p, rel_q, xp.ones(1) * 0.3])
