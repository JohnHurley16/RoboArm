"""Wrist roll + servo-driven parallel gripper with Fin Ray fingertips.

Hand frame: J4 axis at the origin, x forward, z up (same as other links).
Grip frame (G): on the roll axis at the front face of the roll flange,
rotates about +x by J5.  Finger linkage lives in G's x/z plane.
"""
import math
import numpy as np
import cadquery as cq

from .params import FIT, CAP, JOINTS, GRIP, STACK, ARM
from . import hardware as hw
from .core import Part, COLORS, T, R, frame, place_all, smooth_slab, rrect, hull_sketch
from .actuator import CAP_SCREW_ANGLES
from . import arm as A

J5 = JOINTS["J5"]
XF = GRIP["yoke_x"]                      # yoke rear face (actuator mount face)
MF = STACK["mount_face"]
YOKE_T = ARM["cheek_t"]
_cp = XF - MF + STACK["cable_plane"]
ROLL_SLAB = (_cp - CAP["sector_t"] / 2, _cp + CAP["sector_t"] / 2)   # hand x
BOSS_END = XF + 22.0
XG = BOSS_END + 0.5 + 8.0                # grip frame origin (flange front face)
RING_R = 27.5
BRG_SEAT = 11.05

LL, LH, PZ, XP = GRIP["link_L"], GRIP["link_h"], GRIP["pivot_z"], GRIP["pivot_x"]
SV = GRIP["servo"]
WALL_N = (-17.8, -13.8)                  # servo-side palm wall (y)
WALL_P = (9.0, 13.0)
LINK1_Y = (-8.0, -3.0)
LINK2_Y = (3.0, 8.0)
FING_Y = (-2.5, 2.5)
HORN_Y = (-8.0, -6.2)

C_W, C_G, C_A = COLORS["printed"], COLORS["printed2"], COLORS["accent"]
C_TPU = (0.13, 0.14, 0.16)


# ------------------------------------------------------- roll-plane tools ---
def roll_rot(wp):
    """local (X, Y, Z) -> hand (y, z, x)."""
    return wp.rotate((0, 0, 0), (1, 1, 1), 120)


def roll_prism(fn, x0, x1):
    return roll_rot(fn(cq.Workplane("XY").workplane(offset=x0)).extrude(x1 - x0))


def roll_smooth(circles, x0, x1, edge=1.5):
    s = hull_sketch(circles)(cq.Workplane("XY").workplane(offset=x0)).extrude(x1 - x0)
    if edge:
        try:
            s2 = s.faces(">Z or <Z").edges().fillet(edge)
            if s2.val().isValid():
                s = s2
        except Exception:
            pass
    return roll_rot(s)


def cyl_x(y, z, r, x0, x1):
    return cq.Workplane("XY").add(cq.Solid.makeCylinder(r, x1 - x0, cq.Vector(x0, y, z),
                                                        cq.Vector(1, 0, 0)))


def frame_x(x, y, z, direction):
    """Local +Z -> (direction, 0, 0)."""
    if direction > 0:
        return frame((0, 1, 0), (0, 0, 1), (1, 0, 0), (x, y, z))
    return frame((0, 1, 0), (0, 0, -1), (-1, 0, 0), (x, y, z))


def roll_act_matrix():
    d, a = J5["d"], math.radians(J5["dir"])
    return frame((0, 1, 0), (0, 0, 1), (1, 0, 0), (XF - MF, d * math.cos(a), d * math.sin(a)))


def roll_pulley_yz():
    a = math.radians(J5["dir"])
    return (J5["d"] * math.cos(a), J5["d"] * math.sin(a))


# ------------------------------------------------------------ hand body -----
NECK = [(0.0, 0.0, A.HUB_R), (14.0, -4.5, 8.5)]
NECK2 = [(14.0, -4.5, 8.5)] + rrect(14.0, -13.0, XF, 4.0, 4.0)   # stays under the roll actuator
NECK_BOLTS = [(10.5, -10.0), (-10.5, -10.0)]                         # (y, z) at x = XF


def hand_body():
    s4 = JOINTS["J4"]["side"]
    A._BOLTS["hand"] = A.sector_bolt_angles(JOINTS["J4"]["arc"], [NECK, NECK2])
    p = A.link_body("hand", s4, NECK, [], extra_circles=NECK2)
    b = p.shape
    for y, z in NECK_BOLTS:
        b = b.cut(cyl_x(y, z, FIT["insert_m3_d"] / 2, XF - FIT["insert_m3_depth"], XF))
    b = b.cut(cyl_x(0, 0, 9.5, XF - 6.0, XF))                   # clearance for shaft end + e-clip
    p.shape = b
    return p


def wrist_yoke():
    py, pz = roll_pulley_yz()
    outline = [(0.0, 0.0, 22.0), (py, pz, 35.0)]
    b = roll_smooth(outline, XF, XF + YOKE_T, edge=1.6)
    b = b.union(cyl_x(0, 0, 15.5, XF + YOKE_T - 0.5, BOSS_END))
    b = b.cut(cyl_x(0, 0, 8.5, XF, XF + 1.0))
    b = b.cut(cyl_x(0, 0, BRG_SEAT, XF + 1.0, BOSS_END))
    b = b.cut(cyl_x(py, pz, A.TUBE_R + 0.5, XF, XF + YOKE_T))
    for p in A.cap_screw_points(roll_act_matrix()):
        b = b.cut(cyl_x(p[1], p[2], FIT["m3_clear"] / 2, XF, XF + YOKE_T))
        b = b.cut(cyl_x(p[1], p[2], FIT["m3_head_d"] / 2, XF + YOKE_T - FIT["m3_head_h"], XF + YOKE_T))
    for y, z in NECK_BOLTS:
        b = b.cut(cyl_x(y, z, FIT["m3_clear"] / 2, XF, XF + YOKE_T))
        b = b.cut(cyl_x(y, z, FIT["m3_head_d"] / 2, XF + YOKE_T - FIT["m3_head_h"], XF + YOKE_T))
    for a in GUARD_BOSSES:
        by, bz = py + GUARD_R * math.cos(math.radians(a)), pz + GUARD_R * math.sin(math.radians(a))
        b = b.cut(cyl_x(by, bz, FIT["insert_m3_d"] / 2, XF + YOKE_T - FIT["insert_m3_depth"], XF + YOKE_T))
    return Part("wrist_yoke", b, "printed", C_G, print_down=(-1, 0, 0))


GUARD = dict(r_in=21.0, r_out=23.0, a0=-25.0, a1=205.0, front=XF - MF + STACK["pul_front"][1] + 0.6)
GUARD_BOSSES = (0.0, 180.0)
GUARD_R = 25.0


def _arc(r0, r1, a0, a1, cy, cz):
    pts = [(cy + r1 * math.cos(math.radians(a)), cz + r1 * math.sin(math.radians(a)))
           for a in np.linspace(a0, a1, 48)]
    pts += [(cy + r0 * math.cos(math.radians(a)), cz + r0 * math.sin(math.radians(a)))
            for a in np.linspace(a1, a0, 48)]
    return pts


def pulley_guard():
    """Shroud over the roll capstan pulley (hides the cable, keeps fingers out)."""
    py, pz = roll_pulley_yz()
    g = GUARD
    x0, x1 = XF + YOKE_T, g["front"] + 1.6
    b = roll_prism(lambda wp: wp.polyline(_arc(g["r_in"], g["r_out"], g["a0"], g["a1"], py, pz)).close(), x0, x1)
    cap = [(py, pz)] + [(py + g["r_out"] * math.cos(math.radians(a)), pz + g["r_out"] * math.sin(math.radians(a)))
                        for a in np.linspace(g["a0"], g["a1"], 48)]
    b = b.union(roll_prism(lambda wp: wp.polyline(cap).close(), g["front"], x1))
    for a in GUARD_BOSSES:
        by, bz = py + GUARD_R * math.cos(math.radians(a)), pz + GUARD_R * math.sin(math.radians(a))
        b = b.union(cyl_x(by, bz, 3.6, x0, x0 + 8.0))
        b = b.cut(cyl_x(by, bz, FIT["m3_clear"] / 2, x0, x1))
        b = b.cut(cyl_x(by, bz, FIT["m3_head_d"] / 2, x0 + 8.0, x1))
    try:
        b = b.faces(">X").edges().chamfer(0.8)
    except Exception:
        pass
    return Part("roll_pulley_guard", b, "printed", C_G, print_down=(1, 0, 0))


def hand_parts():
    P = []
    s4 = JOINTS["J4"]["side"]
    P.append(hand_body())
    P += A.body_hardware("hand", s4, [])
    P.append(A.link_sector("sector_j4", s4, JOINTS["J4"]["Rc"], JOINTS["J4"]["arc"], [NECK, NECK2]))
    P += A.sector_hardware(s4, JOINTS["J4"]["Rc"], JOINTS["J4"]["arc"], "j4")
    P += A.joint_shaft_parts("j4")
    for y, z in NECK_BOLTS:
        P.append(A.hwpart("ins_m3", hw.heat_insert("M3"), frame_x(XF, y, z, 1), A.INSERT_BOM, COLORS["brass"]))
        P.append(A.hwpart("shcs_m3x10", hw.shcs("M3", 10), frame_x(XF + YOKE_T - FIT["m3_head_h"], y, z, 1),
                          "M3x10 socket head"))
    P.append(wrist_yoke())
    P.append(pulley_guard())
    py, pz = roll_pulley_yz()
    for a in GUARD_BOSSES:
        by, bz = py + GUARD_R * math.cos(math.radians(a)), pz + GUARD_R * math.sin(math.radians(a))
        P.append(A.hwpart("ins_m3", hw.heat_insert("M3"), frame_x(XF + YOKE_T, by, bz, -1), A.INSERT_BOM,
                          COLORS["brass"]))
        P.append(A.hwpart("shcs_m3x12", hw.shcs("M3", 12), frame_x(XF + YOKE_T + 8.0, by, bz, 1),
                          "M3x12 socket head"))
    M = roll_act_matrix()
    P += A.actuator_in(M, "j5")
    for p in A.cap_screw_points(M):
        P.append(A.hwpart("shcs_m3x30", hw.shcs("M3", 30), frame_x(XF + YOKE_T - FIT["m3_head_h"], p[1], p[2], 1),
                          "M3x30 socket head (actuator through-bolt)"))
    for x in (XF + 1.0, BOSS_END - 7.0):
        P.append(Part("brg_608", hw.bearing("608"), "hardware", COLORS["bearing"],
                      M=frame_x(x, 0, 0, 1), bom="608 bearing (8x22x7)"))
    P.append(Part("roll_spacer", hw.ring(11.0, 8.3, BOSS_END - 7.0 - (XF + 8.0)), "printed", C_G,
                  M=frame_x(XF + 8.0, 0, 0, 1), print_down=(0, 0, -1)))
    return P


# --------------------------------------------------------- roll flange ------
def gx(x_hand):
    return x_hand - XG


def roll_flange():
    """Capstan sector for the roll + flange the palm bolts to (grip frame)."""
    Rc, arc = J5["Rc"], J5["arc"]
    s0, s1 = gx(ROLL_SLAB[0]), gx(ROLL_SLAB[1])
    disc0 = gx(BOSS_END + 0.5)
    b = roll_prism(lambda wp: wp.circle(RING_R), s0, 0.0)
    b = b.union(roll_prism(lambda wp: wp.polyline(A.wedge_pts(Rc, arc, RING_R - 1)).close(), s0, s1))
    b = b.cut(cyl_x(0, 0, 16.2, s0, disc0))
    b = b.cut(cyl_x(0, 0, A.SHAFT_BORE, disc0, 0.0))
    for f in A.sector_features(Rc, arc):
        b = b.cut(roll_prism(lambda wp, f=f: wp.polyline(A.notch_pts(Rc, f["notch"])).close(), s0, s1))
        u, v = f["clamp"]
        b = b.cut(cyl_x(u, v, FIT["insert_m3_d"] / 2, s0, s0 + FIT["insert_m3_depth"]))
    for y, z in PALM_BOLTS:
        b = b.cut(cyl_x(y, z, FIT["insert_m3_d"] / 2, -FIT["insert_m3_depth"], 0.0))
    # radial set screw from below (sits in the sector's gap)
    zc = (disc0 + 0.0) / 2
    b = b.cut(cq.Workplane("XY").add(cq.Solid.makeCylinder(FIT["insert_m3_d"] / 2, FIT["insert_m3_depth"],
                                                           cq.Vector(zc, 0, -RING_R), cq.Vector(0, 0, 1))))
    b = b.cut(cq.Workplane("XY").add(cq.Solid.makeCylinder(1.6, RING_R, cq.Vector(zc, 0, -RING_R),
                                                           cq.Vector(0, 0, 1))))
    try:
        b = b.faces(">X").edges(cq.selectors.RadiusNthSelector(-1)).chamfer(1.0)
    except Exception:
        pass
    return Part("roll_flange", b, "printed", C_A, print_down=(-1, 0, 0))


def roll_flange_hardware():
    Rc, arc = J5["Rc"], J5["arc"]
    s0, s1 = gx(ROLL_SLAB[0]), gx(ROLL_SLAB[1])
    feats = A.sector_features(Rc, arc)
    P = A.clamp_hardware([frame_x(s0, f["clamp"][0], f["clamp"][1], -1) for f in feats])
    P.append(Part("cable_j5", hw.cable_arc(Rc, feats[0]["notch"], feats[1]["notch"], 0.0, CAP["cable_d"]),
                  "cable", COLORS["cable"], M=frame((0, 1, 0), (0, 0, 1), (1, 0, 0), ((s0 + s1) / 2, 0, 0))))
    disc0 = gx(BOSS_END + 0.5)
    shaft_len = 0.0 - 1.0 - gx(XF)
    P.append(A.hwpart("shaft8_j5", hw.shaft(8, shaft_len), frame_x(gx(XF), 0, 0, 1),
                      f"8 mm steel shaft x {shaft_len:.0f} mm", COLORS["steel"]))
    P.append(A.hwpart("eclip_8", hw.e_clip(8), frame_x(gx(XF) + 0.3, 0, 0, 1), "8 mm e-clip (DIN 6799)",
                      COLORS["steel"]))
    P.append(A.hwpart("shim_m8", hw.ring(14, 8.2, 0.5), frame_x(disc0 - 0.5, 0, 0, 1),
                      "M8 shim washer 8x14x0.5", COLORS["steel"]))
    zc = disc0 / 2
    P.append(A.hwpart("ins_m3", hw.heat_insert("M3"), frame((1, 0, 0), (0, -1, 0), (0, 0, -1), (zc, 0, -RING_R)),
                      A.INSERT_BOM, COLORS["brass"]))
    P.append(A.hwpart("grub_m3x8", cq.Workplane("XY").circle(1.5).extrude(8),
                      T(zc, 0, -4.0 - 8.0),
                      "M3x8 grub screw (shaft lock)"))
    for y, z in PALM_BOLTS:
        P.append(A.hwpart("ins_m3", hw.heat_insert("M3"), frame_x(0.0, y, z, -1), A.INSERT_BOM, COLORS["brass"]))
    return P


# ---------------------------------------------------------------- palm ------
PALM_PLATE = 6.0
PALM_BOLTS = [(2.0, 15.5), (2.0, -15.5), (-11.0, 0.0)]
P1U, P2U, P1L, P2L = (XP, PZ), (XP, PZ + LH), (XP, -PZ), (XP, -PZ - LH)
SERVO_CX = XP - (SV["body"][0] / 2 - SV["shaft_off"])
TAB_HOLES = [(SERVO_CX - SV["tab_holes"] / 2, PZ), (SERVO_CX + SV["tab_holes"] / 2, PZ)]


def palm():
    lug = 6.5
    wall_c = rrect(0.0, -26.0, 10.0, 26.0, 5.0) + [(XP, z, lug) for z in (PZ, PZ + LH, -PZ, -PZ - LH)]
    b = roll_prism(lambda wp: wp.circle(26.0), 0.0, PALM_PLATE)
    try:
        b = b.faces("<X").edges().chamfer(1.5)
    except Exception:
        pass
    b = b.union(smooth_slab(wall_c + [(x, z, 4.5) for x, z in TAB_HOLES], *WALL_N, edge=1.2))
    b = b.union(smooth_slab(wall_c, *WALL_P, edge=1.2))
    for zs in (1, -1):
        z0, z1 = sorted((zs * 19.0, zs * 26.0))
        b = b.union(smooth_slab(rrect(2.0, z0, XP - 8.0, z1, 3.0), WALL_N[1] - 0.5, WALL_P[0] + 0.5, edge=0))
    b = b.union(smooth_slab(rrect(2.0, -12.0, XP - 12.0, 12.0, 3.0), -8.0, WALL_P[0] + 0.5, edge=0))
    # servo window in the servo-side wall
    bx, bz, _ = SV["body"]
    sx0, sx1 = XP - (bx - SV["shaft_off"]) - 0.4, XP + SV["shaft_off"] + 0.4
    b = b.cut(A.poly_slab([(sx0, PZ - bz / 2 - 0.4), (sx1, PZ - bz / 2 - 0.4),
                           (sx1, PZ + bz / 2 + 0.4), (sx0, PZ + bz / 2 + 0.4)], WALL_N[0] - 1, WALL_N[1] + 1))
    for x, z in TAB_HOLES:
        b = b.cut(A.circle_slab(x, z, 0.9, *WALL_N))
    for x, z in (P2U, P1L, P2L):
        b = b.cut(A.circle_slab(x, z, 1.65, WALL_N[0], WALL_P[1]))
    b = b.cut(A.circle_slab(P1U[0], P1U[1], 3.0, *WALL_P))         # screwdriver access to the horn screw
    for y, z in PALM_BOLTS:
        b = b.cut(cyl_x(y, z, FIT["m3_clear"] / 2, 0.0, PALM_PLATE))
        b = b.cut(cyl_x(y, z, FIT["m3_head_d"] / 2, PALM_PLATE - FIT["m3_head_h"], PALM_PLATE + 6))
    return Part("palm", b, "printed", C_W, print_down=(-1, 0, 0))


def servo_cover():
    bx, bz, bh = SV["body"]
    y_in = WALL_N[0]
    body_bottom = HORN_Y[1] - SV["spline_h"] - SV["boss_h"] - bh
    y_out = body_bottom - 1.6
    x0, x1 = SERVO_CX - SV["tab_len"] / 2 - 1.5, SERVO_CX + SV["tab_len"] / 2 + 1.5
    c = smooth_slab(rrect(x0, PZ - bz / 2 - 2.0, x1, PZ + bz / 2 + 2.0, 4.0), y_out, y_in, edge=2.0)
    tab_bot = y_in - (SV["tab_h"][1] - SV["tab_h"][0]) - 0.2
    c = c.cut(A.poly_slab([(SERVO_CX - SV["tab_len"] / 2 - 0.3, PZ - bz / 2 - 0.3),
                           (SERVO_CX + SV["tab_len"] / 2 + 0.3, PZ - bz / 2 - 0.3),
                           (SERVO_CX + SV["tab_len"] / 2 + 0.3, PZ + bz / 2 + 0.3),
                           (SERVO_CX - SV["tab_len"] / 2 - 0.3, PZ + bz / 2 + 0.3)], tab_bot, y_in))
    c = c.cut(A.poly_slab([(SERVO_CX - bx / 2 - 0.3, PZ - bz / 2 - 0.3), (SERVO_CX + bx / 2 + 0.3, PZ - bz / 2 - 0.3),
                           (SERVO_CX + bx / 2 + 0.3, PZ + bz / 2 + 0.3), (SERVO_CX - bx / 2 - 0.3, PZ + bz / 2 + 0.3)],
                          body_bottom - 0.3, y_in))
    for x, z in TAB_HOLES:
        c = c.cut(A.circle_slab(x, z, 1.2, y_out, y_in))
        c = c.cut(A.circle_slab(x, z, 2.2, y_out, y_out + 2.0))
    # lead exit at the rear end
    c = c.cut(A.poly_slab([(x0 - 1, PZ - 3.5), (SERVO_CX - bx / 2 + 0.5, PZ - 3.5),
                           (SERVO_CX - bx / 2 + 0.5, PZ + 3.5), (x0 - 1, PZ + 3.5)], body_bottom + 1.0, body_bottom + 7.0))
    return Part("servo_cover", c, "printed", C_W, print_down=(0, -1, 0))


def servo_matrix():
    return frame((1, 0, 0), (0, 0, -1), (0, 1, 0), (XP, HORN_Y[1], PZ))


def palm_hardware():
    P = []
    M = servo_matrix()
    for k, s in hw.micro_servo(SV).items():
        P.append(Part(k, s, "motor", (0.12, 0.13, 0.16) if k == "servo_body" else COLORS["steel"], M=M,
                      bom=SV["name"] if k == "servo_body" else None))
    for x, z in TAB_HOLES:
        P.append(A.hwpart("screw_m2x20", hw.shcs("M2", 20),
                          A.frame_y(x, z, servo_cover_out() + 2.0, -1), "M2x20 self-tapping screw (servo)"))
    for x, z in (P2U, P1L, P2L):
        P.append(A.hwpart("shcs_m3x35", hw.shcs("M3", 35), A.frame_y(x, z, WALL_P[1], 1),
                          "M3x35 socket head (palm pivot)"))
        P.append(A.hwpart("nyloc_m3", hw.nyloc("M3"), A.frame_y(x, z, WALL_N[0], -1), "M3 nyloc nut"))
    for y, z in PALM_BOLTS:
        P.append(A.hwpart("shcs_m3x8", hw.shcs("M3", 8), frame_x(PALM_PLATE - FIT["m3_head_h"], y, z, 1),
                          "M3x8 socket head"))
    return P


def servo_cover_out():
    bh = SV["body"][2]
    return HORN_Y[1] - SV["spline_h"] - SV["boss_h"] - bh - 1.6


# -------------------------------------------------------------- linkage -----
def link_drive(upper=True):
    zt = 12
    r_p = GRIP["gear_module"] * zt / 2
    assert abs(r_p - PZ) < 1e-6, "sync gear pitch radius must equal pivot_z"
    phase = -90.0 if upper else 90.0 + 180.0 / zt
    b = A.smooth_slab([(0.0, 0.0, 5.0), (LL, 0.0, 4.0)], *LINK1_Y, edge=0.6)
    b = b.union(A.poly_slab(A.involute_gear_pts(GRIP["gear_module"], zt, phase), *LINK1_Y))
    b = b.cut(A.circle_slab(LL, 0, 1.65, *LINK1_Y))
    if upper:
        horn = hw.servo_horn()
        pts = [(0, -2.8), (15.2, -2.0), (15.2, 2.0), (0, 2.8)]
        b = b.cut(A.circle_slab(0, 0, 3.8, *HORN_Y))
        b = b.cut(A.poly_slab(pts, *HORN_Y))
        b = b.cut(A.circle_slab(0, 0, 2.1, HORN_Y[1], LINK1_Y[1]))
        return Part("link_drive_upper", b, "printed", C_G, print_down=(0, 1, 0))
    b = b.cut(A.circle_slab(0, 0, 1.65, *LINK1_Y))
    return Part("link_drive_lower", b, "printed", C_G, print_down=(0, 1, 0))


def link_follow():
    b = A.smooth_slab([(0.0, 0.0, 4.5), (LL, 0.0, 4.0)], *LINK2_Y, edge=0.6)
    b = b.cut(A.circle_slab(0, 0, 1.65, *LINK2_Y)).cut(A.circle_slab(LL, 0, 1.65, *LINK2_Y))
    return Part("link_follow", b, "printed", C_G, print_down=(0, -1, 0))


FACE = GRIP["face"]
TONGUE = (6.0, -FACE, 20.0, 4.0)
FIN_BASE = (8.0, 20.0)
FIN_TIP = 52.0
FIN_Y = 7.0
FIN_TOP = 9.0


def finger():
    """Rigid finger (upper).  Frame at the lower pin F1; pad face at z=-FACE."""
    b = A.smooth_slab([(0.0, 0.0, 4.5), (0.0, LH, 4.5)] + rrect(-3.0, -FACE, 7.0, LH + 2.0, 2.5), *FING_Y, edge=0.6)
    b = b.union(A.smooth_slab(rrect(*TONGUE, 1.5), *FING_Y, edge=0))
    for x, z in ((0.0, 0.0), (0.0, LH)):
        b = b.cut(A.circle_slab(x, z, 1.65, *FING_Y))
    for x in (11.0, 17.0):
        b = b.cut(A.circle_slab(x, -0.5, 1.7, *FING_Y))
    return Part("finger", b, "printed", C_W, print_down=(0, -1, 0))


def fin_ray():
    """TPU Fin Ray fingertip: slips over the finger tongue, two M3 screws."""
    x0, x1 = FIN_BASE
    base = A.poly_slab([(x0, -FACE), (x1, -FACE), (x1, FIN_TOP), (x0, FIN_TOP)], -FIN_Y, FIN_Y)
    tip_t = 1.8
    outer = lambda x: FIN_TOP + (-FACE + tip_t - FIN_TOP) * (x - x1) / (FIN_TIP - x1)
    fin = A.poly_slab([(x1, -FACE), (FIN_TIP, -FACE), (FIN_TIP, -FACE + tip_t), (x1, FIN_TOP)], -FIN_Y, FIN_Y)
    b = base.union(fin)
    ribs = np.linspace(x1, FIN_TIP - 4, 7)
    for a, c in zip(ribs[:-1], ribs[1:]):
        xa, xb = a + 0.7, c - 0.7
        za0 = -FACE + 1.4
        za1, zb1 = outer(xa) - 1.4, outer(xb) - 1.4
        if min(za1, zb1) - za0 > 1.0:
            b = b.cut(A.poly_slab([(xa, za0), (xb, za0), (xb, zb1), (xa, za1)], -FIN_Y - 1, FIN_Y + 1))
    # tongue slot + screw holes
    tx0, tz0, tx1, tz1 = TONGUE
    b = b.cut(A.poly_slab([(x0 - 1, tz0 - 0.3), (tx1 + 0.3, tz0 - 0.3), (tx1 + 0.3, tz1 + 0.3), (x0 - 1, tz1 + 0.3)],
                          FING_Y[0] - 0.25, FING_Y[1] + 0.25))
    for x in (11.0, 17.0):
        b = b.cut(A.circle_slab(x, -0.5, 1.7, -FIN_Y, FIN_Y))
        b = b.cut(A.circle_slab(x, -0.5, 3.2, FIN_Y - 2.4, FIN_Y))
    # grip texture on the contact face
    for i in range(10):
        x = x1 + 2.0 + i * 3.0
        if x > FIN_TIP - 3:
            break
        b = b.cut(A.poly_slab([(x, -FACE - 0.1), (x + 1.0, -FACE - 0.1), (x + 1.0, -FACE + 0.5), (x, -FACE + 0.5)],
                              -FIN_Y - 1, FIN_Y + 1))
    return Part("finray_tpu", b, "printed", C_TPU, print_down=(0, -1, 0))


def finger_hardware(flip=False):
    """Pins + nuts for a finger.  flip=True for the lower finger, whose frame is
    rotated 180 deg about x (its local +y is world -y)."""
    f = -1 if flip else 1
    fy = lambda x, z, y, d: A.frame_y(x, z, f * y, f * d)
    P = []
    P.append(A.hwpart("shcs_m3x12", hw.shcs("M3", 12), fy(0, 0, LINK1_Y[0], -1), "M3x12 socket head (finger pin)"))
    P.append(A.hwpart("nyloc_m3", hw.nyloc("M3"), fy(0, 0, FING_Y[1], 1), "M3 nyloc nut"))
    P.append(A.hwpart("shcs_m3x12", hw.shcs("M3", 12), fy(0, LH, LINK2_Y[1], 1), "M3x12 socket head (finger pin)"))
    P.append(A.hwpart("nyloc_m3", hw.nyloc("M3"), fy(0, LH, FING_Y[0], -1), "M3 nyloc nut"))
    for x in (11.0, 17.0):
        P.append(A.hwpart("shcs_m3x16", hw.shcs("M3", 16), A.frame_y(x, -0.5, FIN_Y - 2.4, 1), "M3x16 socket head (fingertip)"))
        P.append(A.hwpart("nyloc_m3", hw.nyloc("M3"), A.frame_y(x, -0.5, -FIN_Y, -1), "M3 nyloc nut"))
    return P


# ---------------------------------------------------------------- build -----
def build():
    bodies = {}
    bodies["hand"] = hand_parts()
    G = [roll_flange()] + roll_flange_hardware() + [palm(), servo_cover()] + palm_hardware()
    bodies["grip"] = G
    l1u = [link_drive(True), Part("servo_horn", hw.servo_horn(), "hardware", (0.92, 0.92, 0.92),
                                  M=A.frame_y(0, 0, HORN_Y[0], 1), bom=None)]
    bodies["l1u"] = l1u
    bodies["l1l"] = [link_drive(False)]
    bodies["l2u"] = [link_follow()]
    bodies["l2l"] = [link_follow()]
    fg, fr = finger(), fin_ray()
    bodies["fu"] = [fg, fr] + finger_hardware(False)
    bodies["fl"] = [fg, fr] + finger_hardware(True)
    return bodies


def frames(Fh, q5, phi):
    """World matrices of every gripper body for roll q5 and link angle phi."""
    Fg = Fh @ T(XG, 0, 0) @ R((1, 0, 0), q5)
    up = lambda deg: R((0, -1, 0), deg)
    c, s = math.cos(math.radians(phi)), math.sin(math.radians(phi))
    return dict(
        grip=Fg,
        l1u=Fg @ T(XP, 0, PZ) @ up(phi),
        l2u=Fg @ T(XP, 0, PZ + LH) @ up(phi),
        l1l=Fg @ T(XP, 0, -PZ) @ up(-phi),
        l2l=Fg @ T(XP, 0, -PZ - LH) @ up(-phi),
        fu=Fg @ T(XP + LL * c, 0, PZ + LL * s),
        fl=Fg @ T(XP + LL * c, 0, -PZ - LL * s) @ R((1, 0, 0), 180),
    )


def opening_mm(phi):
    """Gap between the two pad faces (rigid, before Fin Ray deflection)."""
    return 2 * (PZ + LL * math.sin(math.radians(phi)) - FACE)
