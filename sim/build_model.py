"""Generate the MuJoCo model of Capstan Arm 5 from the CAD.

Writes sim/assets/arm.xml (+ visual meshes in sim/assets/meshes/) and
sim/assets/arm_meta.json (masses, motor data, balancer curve) that the
environments read.  Needs the CAD build first (python cad/build.py).  Run from the repo root:

    python sim/build_model.py

Body frames are identical to the CAD's (arm/arm.py body_frames), so joint
angles mean the same thing here, in the IK (ik/capstan_ik.py) and on the
real arm.  Units: metres, kg, seconds, radians.
"""
import json
import math
import os
import sys

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CAD = os.path.join(ROOT, "cad")                 # the parametric CAD package lives in cad/arm
sys.path.insert(0, CAD)

from arm.arm import build_static, J2_Z, L1, L2                      # noqa: E402
from arm import hand                                                  # noqa: E402
from arm.core import COLORS                                           # noqa: E402
from arm.massprops import part_mass                                   # noqa: E402
from arm.params import JOINTS                                         # noqa: E402
from arm.render import mesh_of                                        # noqa: E402

ASSETS = os.path.join(HERE, "assets")
MESHES = os.path.join(ASSETS, "meshes")
MM = 1e-3

# ----------------------------------------------------------- motor data -----
# CubeMars datasheets (cubemars.com, GL60 KV25 and GL40 KV70 product pages).
# Resistance and KV are phase-to-phase values; used as a DC-equivalent model.
MOTOR = {
    "GL60": dict(kv_rpm=21.5, R=5.5, kt=0.444, i_peak=4.0, i_rated=1.35, tau_peak=1.75, tau_rated=0.6,
                 rotor_inertia=355e-7),     # kg m^2  (355 g cm^2)
    "GL40": dict(kv_rpm=63.5, R=4.5, kt=0.152, i_peak=3.3, i_rated=1.62, tau_peak=0.5, tau_rated=0.25,
                 rotor_inertia=74e-7),
}
V_BUS = 24.0
CAPSTAN_EFF = 0.92
# MG90S micro servo: 0.1 s / 60 deg at 4.8 V, ~0.21 N m stall (on the drive link directly)
SERVO = dict(speed_dps=600.0, stall_nm=0.21, latency_s=0.015, latency_jitter_s=0.003)

# collision bits.  Arm links only touch the world; the hand also touches the bag.
ARM, WORLD, BAG, BASE, HAND, FABRIC, PAD = 1, 2, 4, 8, 16, 32, 64
# The Fin Ray pads (PAD) touch the bag's fabric (FABRIC), not its beans: a pinch squeezes the
# beans aside and clamps cloth.  Other hand parts (HAND) push on the beans normally.
HAND_BODIES = ("grip", "l1u", "l1l", "fu", "fl")   # the parts that can touch a held bag

PALETTE = {   # visual material per colour group
    "white": COLORS["printed"], "graphite": COLORS["printed2"], "orange": COLORS["accent"],
    "black": (0.07, 0.07, 0.08), "steel": COLORS["steel"],
}
HIDDEN_KINDS = {"motor", "pcb"}
HIDDEN_KEYS = ("brg_", "ins_", "motor_", "magnet", "drv_", "act_spider", "act_retainer", "act_bearing_spacer",
               "act_magnet_plug", "act_rotor_adapter", "balancer_spring", "servo_body", "servo_spline")


def colour_of(p):
    c = np.asarray(p.color)
    if p.kind == "printed":
        best = min((("white", COLORS["printed"]), ("graphite", COLORS["printed2"]),
                    ("orange", COLORS["accent"]), ("black", (0.13, 0.14, 0.16))),
                   key=lambda s: np.linalg.norm(c - np.asarray(s[1])))
        return best[0]
    if np.allclose(c, COLORS["screw"], atol=0.02):
        return "black"
    return "steel"


def visible(p):
    return p.kind not in HIDDEN_KINDS and p.kind != "cable" and not p.key.startswith(HIDDEN_KEYS)


# ------------------------------------------------------------- helpers ------
_mesh_cache = {}


def part_mesh(p, fine=False):
    k = (p.key, id(p.shape), fine)
    if k not in _mesh_cache:
        V, F = mesh_of(p.local_shape(), tol=0.05 if fine else 0.25, ang=0.2 if fine else 0.5)
        m = trimesh.Trimesh(V, F, process=True)
        _mesh_cache[k] = m
    m = _mesh_cache[k].copy()
    m.apply_transform(p.M)
    return m


def box_inertia(ext, m):
    a, b, c = ext
    return m / 12.0 * np.diag([b * b + c * c, a * a + c * c, a * a + b * b])


def body_inertial(parts):
    """mass (kg), com (m), full inertia tensor about com (kg m^2), body frame."""
    items = []
    for p in parts:
        if p.kind == "cable":
            continue
        m = part_mass(p) / 1000.0
        if m <= 0:
            continue
        mesh = part_mesh(p, fine=True)
        if mesh.is_volume and mesh.volume > 1e-6:
            c = mesh.center_mass * MM
            I = mesh.moment_inertia / mesh.mass * m * MM * MM          # density-free, scaled to the part mass
        else:
            obb = mesh.bounding_box_oriented
            c = obb.primitive.transform[:3, 3] * MM
            R = obb.primitive.transform[:3, :3]
            I = R @ box_inertia(obb.primitive.extents * MM, m) @ R.T
        items.append((m, c, I))
    M = sum(m for m, _, _ in items)
    com = sum(m * c for m, c, _ in items) / M
    I = np.zeros((3, 3))
    for m, c, Ii in items:
        d = c - com
        I += Ii + m * (d @ d * np.eye(3) - np.outer(d, d))
    return M, com, I


def fullinertia(I):
    return " ".join(f"{v:.6e}" for v in (I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]))


# collision proxies: the big printed shells and actuator cans, as oriented boxes
COLLIDE = ("turret_body", "turret_drum", "upper_arm", "upper_arm_cheek", "forearm", "forearm_cheek",
           "act_motor_can", "act_front_cap", "hand", "wrist_yoke", "roll_flange", "palm", "servo_cover",
           "balancer_tube", "balancer_bracket", "sector_j", "link_drive", "finger")


def collision_boxes(parts):
    out = []
    for p in parts:
        if p.kind != "printed" or not p.key.startswith(COLLIDE):
            continue
        mesh = part_mesh(p)
        obb = mesh.bounding_box_oriented
        T = obb.primitive.transform
        out.append((p.key, T[:3, 3] * MM, T[:3, :3], obb.primitive.extents * MM / 2))
    return out


def quat_of(R):
    q = trimesh.transformations.quaternion_from_matrix(np.r_[np.c_[R, [0, 0, 0]], [[0, 0, 0, 1]]])
    return " ".join(f"{v:.6f}" for v in q)    # w x y z


# ------------------------------------------------------------- build --------
def main():
    os.makedirs(MESHES, exist_ok=True)
    for f in os.listdir(MESHES):
        os.remove(os.path.join(MESHES, f))
    print("building CAD ...")
    B = build_static()
    print("meshing and mass properties ...")
    meshes_xml, geoms, inertial = [], {}, {}
    for body, parts in B.items():
        # visual meshes, one per colour
        groups = {}
        for p in parts:
            if visible(p):
                groups.setdefault(colour_of(p), []).append(part_mesh(p))
        g = []
        for col, ms in groups.items():
            m = trimesh.util.concatenate(ms)
            m.apply_scale(MM)
            name = f"{body}_{col}"
            m.export(os.path.join(MESHES, name + ".stl"))
            meshes_xml.append(f'    <mesh name="{name}" file="meshes/{name}.stl"/>')
            g.append(f'<geom class="visual" mesh="{name}" material="{col}"/>')
        # collisions
        for key, c, R, h in collision_boxes(parts):
            cls = ("pad" if key == "finger" else "base_col" if body == "base"
                   else "hand_col" if body in HAND_BODIES else "arm_col")
            g.append(f'<geom class="{cls}" type="box" pos="{c[0]:.5f} {c[1]:.5f} {c[2]:.5f}" quat="{quat_of(R)}" '
                     f'size="{h[0]:.5f} {h[1]:.5f} {h[2]:.5f}"/>')
        geoms[body] = g
        inertial[body] = body_inertial(parts)
        print(f"  {body:7s} {inertial[body][0] * 1000:7.1f} g   visual parts {sum(len(v) for v in groups.values())}")

    # Fin Ray pads: soft TPU contact boxes on the inner face of each finger (finger frame)
    fx0, fx1 = hand.FIN_BASE[0], hand.FIN_TIP if hasattr(hand, "FIN_TIP") else 52.0
    pad = ((fx0 + fx1) / 2 * MM, 0.0, (-hand.FACE + 3.0) * MM, (fx1 - fx0) / 2 * MM, 7.0 * MM, 3.0 * MM)
    for b in ("fu", "fl"):
        geoms[b].append(f'<geom name="pad_{b}" class="pad" type="box" pos="{pad[0]:.5f} 0 {pad[2]:.5f}" '
                        f'size="{pad[3]:.5f} {pad[4]:.5f} {pad[5]:.5f}"/>')

    # base: octagonal prism from two boxes (static, collides with the bag only)
    from arm.arm import BASE_R, Hb
    r = BASE_R * MM * math.cos(math.pi / 8)
    for a in (0.0, 45.0):
        q = trimesh.transformations.quaternion_about_axis(math.radians(a), (0, 0, 1))
        geoms["base"] = [g for g in geoms["base"] if 'class="base_col"' not in g or "octa" in g]
        geoms["base"].append(f'<geom name="base_octa_{int(a)}" class="base_col" type="box" pos="0 0 {Hb * MM / 2:.4f}" '
                             f'quat="{" ".join(f"{v:.6f}" for v in q)}" size="{r:.4f} {r:.4f} {Hb * MM / 2:.4f}"/>')

    # armature (reflected rotor inertia) and motor tables
    act = {}
    for j in ("J1", "J2", "J3", "J4", "J5"):
        mot = MOTOR[JOINTS[j]["motor"]]
        n = JOINTS[j]["ratio"]
        act[j] = dict(motor=JOINTS[j]["motor"], ratio=n, armature=mot["rotor_inertia"] * n * n,
                      rom=[math.radians(v) for v in JOINTS[j]["rom"]], **mot)

    def inert(body):
        m, c, I = inertial[body]
        return (f'<inertial pos="{c[0]:.5f} {c[1]:.5f} {c[2]:.5f}" mass="{m:.5f}" '
                f'fullinertia="{fullinertia(I)}"/>')

    def jnt(j, axis):
        a = act[j]
        return (f'<joint name="{j}" axis="{axis}" range="{a["rom"][0]:.5f} {a["rom"][1]:.5f}" '
                f'armature="{a["armature"]:.6f}" damping="0.02" frictionloss="0.03"/>')

    XG, XP, PZ, LH, LL = hand.XG * MM, hand.XP * MM, hand.PZ * MM, hand.LH * MM, hand.LL * MM
    gmax = math.radians(52.0)
    i = lambda n: "  " * n
    G = lambda b, n: "\n".join(i(n) + g for g in geoms[b])
    xml = f"""<mujoco model="capstan_arm_5">
  <!-- generated by sim/build_model.py from the CAD; do not edit by hand -->
  <compiler angle="radian" meshdir="." autolimits="true"/>
  <visual><global offwidth="1920" offheight="1080"/><quality shadowsize="4096"/></visual>
  <default>
    <default class="visual"><geom type="mesh" contype="0" conaffinity="0" group="1" density="0"/></default>
    <default class="arm_col"><geom contype="{ARM}" conaffinity="0" group="3" density="0" rgba="0.3 0.6 1 0.3"/></default>
    <default class="hand_col"><geom contype="{HAND}" conaffinity="0" group="3" density="0" rgba="0.3 1 0.6 0.3"/></default>
    <default class="base_col"><geom contype="{BASE}" conaffinity="0" group="3" density="0" rgba="0.3 0.6 1 0.3"/></default>
    <!-- Fin Ray TPU pads: soft (several mm of give) so they wrap around what they grip, and they win
         the contact-parameter mix (priority) so a pinched beanbag gets the pads' softness -->
    <default class="pad"><geom contype="{PAD}" conaffinity="0" group="3" density="0" friction="1.2 0.02 0.001"
        priority="1" solref="0.008 1" solimp="0.9 0.97 0.002" rgba="1 0.3 0.3 0.4"/></default>
  </default>
  <asset>
    <material name="white" rgba="{PALETTE['white'][0]:.3f} {PALETTE['white'][1]:.3f} {PALETTE['white'][2]:.3f} 1" specular="0.3"/>
    <material name="graphite" rgba="{PALETTE['graphite'][0]:.3f} {PALETTE['graphite'][1]:.3f} {PALETTE['graphite'][2]:.3f} 1" specular="0.3"/>
    <material name="orange" rgba="{PALETTE['orange'][0]:.3f} {PALETTE['orange'][1]:.3f} {PALETTE['orange'][2]:.3f} 1" specular="0.4"/>
    <material name="black" rgba="0.07 0.07 0.08 1" specular="0.2"/>
    <material name="steel" rgba="0.62 0.64 0.68 1" specular="0.6"/>
{chr(10).join(meshes_xml)}
  </asset>
  <worldbody>
    <body name="base" pos="0 0 0">
      {inert('base')}
{G('base', 3)}
      <site name="base_site" pos="0 0 0"/>
      <body name="turret">
        {inert('turret')}
        {jnt('J1', '0 0 1')}
{G('turret', 4)}
        <body name="upper" pos="0 0 {J2_Z * MM:.5f}">
          {inert('upper')}
          {jnt('J2', '0 -1 0')}
{G('upper', 5)}
          <site name="elbow_site" pos="{L1 * MM:.5f} 0 0"/>
          <body name="fore" pos="{L1 * MM:.5f} 0 0">
            {inert('fore')}
            {jnt('J3', '0 -1 0')}
{G('fore', 6)}
            <body name="hand" pos="{L2 * MM:.5f} 0 0">
              {inert('hand')}
              {jnt('J4', '0 -1 0')}
{G('hand', 7)}
              <body name="grip" pos="{XG:.5f} 0 0">
                {inert('grip')}
                {jnt('J5', '1 0 0')}
{G('grip', 8)}
                <body name="l1u" pos="{XP:.5f} 0 {PZ:.5f}">
                  {inert('l1u')}
                  <joint name="G" axis="0 -1 0" range="0 {gmax:.4f}" damping="0.002" armature="0.0001"/>
{G('l1u', 9)}
                  <body name="fu_knuckle" pos="{LL:.5f} 0 0">
                    <joint name="Gku" axis="0 -1 0" range="{-gmax:.4f} 0"/>
                    <body name="fu">
                      {inert('fu')}
{G('fu', 11)}
                      <site name="pad_u" pos="0.030 0 {-hand.FACE * MM:.5f}"/>
                    </body>
                  </body>
                </body>
                <body name="l1l" pos="{XP:.5f} 0 {-PZ:.5f}">
                  {inert('l1l')}
                  <joint name="Gl" axis="0 -1 0" range="{-gmax:.4f} 0"/>
{G('l1l', 9)}
                  <body name="fl_knuckle" pos="{LL:.5f} 0 0">
                    <joint name="Gkl" axis="0 -1 0" range="0 {gmax:.4f}"/>
                    <body name="fl" quat="0 1 0 0">
                      {inert('fl')}
{G('fl', 11)}
                      <site name="pad_l" pos="0.030 0 {-hand.FACE * MM:.5f}"/>
                    </body>
                  </body>
                </body>
                <body name="l2u" pos="{XP:.5f} 0 {PZ + LH:.5f}">
                  {inert('l2u')}
                  <joint name="Gfu" axis="0 -1 0" range="0 {gmax:.4f}"/>
{G('l2u', 9)}
                </body>
                <body name="l2l" pos="{XP:.5f} 0 {-PZ - LH:.5f}">
                  {inert('l2l')}
                  <joint name="Gfl" axis="0 -1 0" range="{-gmax:.4f} 0"/>
{G('l2l', 9)}
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <equality>
    <joint joint1="Gl" joint2="G" polycoef="0 -1 0 0 0" solref="0.004 1"/>
    <joint joint1="Gku" joint2="G" polycoef="0 -1 0 0 0" solref="0.004 1"/>
    <joint joint1="Gkl" joint2="G" polycoef="0 1 0 0 0" solref="0.004 1"/>
    <joint joint1="Gfu" joint2="G" polycoef="0 1 0 0 0" solref="0.004 1"/>
    <joint joint1="Gfl" joint2="G" polycoef="0 -1 0 0 0" solref="0.004 1"/>
  </equality>
  <actuator>
    <!-- joint torque in N m (the env turns policy actions into these, with the motor's torque-speed limit) -->
    <motor name="J1" joint="J1" ctrlrange="-5 5"/>
    <motor name="J2" joint="J2" ctrlrange="-18 18"/>
    <motor name="J3" joint="J3" ctrlrange="-14 14"/>
    <motor name="J4" joint="J4" ctrlrange="-4 4"/>
    <motor name="J5" joint="J5" ctrlrange="-4 4"/>
    <!-- gripper servo: position servo on the drive link, torque limited to the MG90S stall -->
    <!-- a hobby servo reaches stall torque within a few degrees of error: kp = stall / 3 deg -->
    <position name="G" joint="G" kp="4.0" kv="0.03" ctrlrange="0 {gmax:.4f}" forcerange="-{SERVO['stall_nm']} {SERVO['stall_nm']}"/>
  </actuator>
</mujoco>
"""
    open(os.path.join(ASSETS, "arm.xml"), "w").write(xml)
    # collision-only copy for training (no visual meshes: smaller and faster to load into MJX)
    lean = "\n".join(l for l in xml.splitlines() if 'class="visual"' not in l and "<mesh " not in l)
    open(os.path.join(ASSETS, "arm_col.xml"), "w").write(lean)
    from arm.params import GRIP
    bal = json.load(open(os.path.join(CAD, "out", "balancer.json")))      # written by cad/build.py
    meta = dict(
        joints=act, v_bus=V_BUS, capstan_eff=CAPSTAN_EFF, servo=SERVO,
        balancer=dict(q2_deg=bal["curve"]["q2"], torque_nm=bal["curve"]["spring"]),
        masses_g={b: inertial[b][0] * 1000 for b in inertial},
        grip=dict(XP=hand.XP, PZ=hand.PZ, LL=hand.LL, FACE=hand.FACE, phi_max=GRIP["phi"][1]),
        bits=dict(ARM=ARM, WORLD=WORLD, BAG=BAG, BASE=BASE, HAND=HAND, FABRIC=FABRIC, PAD=PAD),
    )
    json.dump(meta, open(os.path.join(ASSETS, "arm_meta.json"), "w"), indent=1)
    import mujoco
    m = mujoco.MjModel.from_xml_path(os.path.join(ASSETS, "arm.xml"))
    print(f"ok: {m.nbody} bodies, {m.njnt} joints, {m.ngeom} geoms, total mass {sum(m.body_mass) * 1000:.0f} g")


if __name__ == "__main__":
    main()
