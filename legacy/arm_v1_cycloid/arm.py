"""Arm structure: base, turret, links, fork cheeks, capstan sectors, gripper.

Link frames: origin on the proximal joint axis, +x along the link, joint
axes along y, link plane = x/z.  Joint angles rotate the child CCW in the
x/z plane (i.e. about -y), so +J2 lifts the arm.
"""
import math
import numpy as np
import cadquery as cq

from .params import (MOTOR, FIT, CYC, CAP, ARM, JOINTS, GRIP, STACK, ENC,
                     SECTOR_Y, CHEEK_OUT, CAP_SCREW_R, TUBE_R, HOUSING_R)
from . import hardware as hw
from .core import (Part, COLORS, T, R, frame, place_all, slab, cyl_y,
                   smooth_slab, floor_slab, rrect, hull_sketch)
from .actuator import actuator_parts, pulley_cable_wrap, CAP_SCREW_ANGLES

IH = ARM["interior_half"]          # 14   fork inner face |y|
CT = ARM["cheek_t"]                # 8
CO = CHEEK_OUT                     # 22   fork outer face |y|
BH = ARM["body_half"]              # 13.5 link body half width
HUB_R = ARM["hub_r"]
PAD_R = ARM["pad_r"]
SEAT_R = hw.BEARINGS["608"][1] / 2 + FIT["bearing_seat"]
SHAFT_BORE = ARM["joint_shaft_d"] / 2 + 0.05
SY0, SY1 = SECTOR_Y                # sector slab |y| (signed by side)
MF = STACK["mount_face"]

C_W = COLORS["printed"]            # white shells / links
C_G = COLORS["printed2"]           # graphite joints
C_A = COLORS["accent"]             # orange capstan parts


def ysort(a, b):
    return (min(a, b), max(a, b))


def deg_at(r, length):
    return math.degrees(length / r)


def pol(cx, cz, r, a):
    return (cx + r * math.cos(math.radians(a)), cz + r * math.sin(math.radians(a)))


def poly_slab(pts, y0, y1):
    return slab(lambda wp: wp.polyline(pts).close(), *ysort(y0, y1))


def circle_slab(cx, cz, r, y0, y1):
    return cyl_y(cx, cz, r, *ysort(y0, y1))


def arc_pts(cx, cz, r, a0, a1, n=None):
    n = n or max(8, int(abs(a1 - a0) / 2.5))
    return [pol(cx, cz, r, a0 + (a1 - a0) * i / n) for i in range(n + 1)]


def kidney_pts(rm, w, a0, a1):
    """Curved window: centreline radius rm, width w, from angle a0 to a1."""
    ro, ri = rm + w / 2, rm - w / 2
    pts = arc_pts(0, 0, ro, a0, a1)
    c1 = pol(0, 0, rm, a1)
    pts += [pol(c1[0], c1[1], w / 2, a1 + t) for t in np.linspace(0, 180, 13)[1:-1]]
    pts += arc_pts(0, 0, ri, a1, a0)
    c0 = pol(0, 0, rm, a0)
    pts += [pol(c0[0], c0[1], w / 2, a0 + 180 + t) for t in np.linspace(0, 180, 13)[1:-1]]
    return pts


def frame_y(x, z, y, direction):
    """Local +Z -> world (0, direction, 0) at link point (x, y, z)."""
    if direction > 0:
        return frame((1, 0, 0), (0, 0, -1), (0, 1, 0), (x, y, z))
    return frame((1, 0, 0), (0, 0, 1), (0, -1, 0), (x, y, z))


def hwpart(key, shape, M, bom, color=None):
    return Part(key, shape, "hardware", color or COLORS["screw"], M=M, bom=bom)


INSERT_BOM = "M3 heat-set insert (M3x5.7, 4.6 OD)"


# ----------------------------------------------------------- actuators ------
def act_matrix_link(px, pz, s, spin=0.0):
    """Actuator frame -> link frame for an actuator on fork cheek side s."""
    if s > 0:
        F = frame((1, 0, 0), (0, 0, 1), (0, -1, 0), (px, s * (CO + MF), pz))
    else:
        F = frame((1, 0, 0), (0, 0, -1), (0, 1, 0), (px, s * (CO + MF), pz))
    return F @ R((0, 0, 1), spin)


def cap_screw_points(M):
    pts = []
    for a in CAP_SCREW_ANGLES:
        v = M @ np.array([CAP_SCREW_R * math.cos(math.radians(a)),
                          CAP_SCREW_R * math.sin(math.radians(a)), MF, 1.0])
        pts.append(v[:3])
    return pts


def actuator_in(M, tag):
    parts = place_all(actuator_parts(), M)
    parts.append(Part("cable_" + tag + "_wrap", pulley_cable_wrap(), "cable",
                      COLORS["cable"]).placed(M))
    return parts


# ------------------------------------------------------------- cheeks -------
def cheek(key, sg, joint=None, pulley=None, act_M=None, lobe=None, bridge=None,
          bridge_screws=(), enc=False, pivots=(), pivot_insert=False, color=C_G,
          teardrop=False):
    """Flat fork cheek on side sg (+1/-1).  Inner face |y|=IH, outer |y|=CO.
    pulley: actuator mounted here (tube hole + screws); lobe: matching round
    lobe without holes (keeps both cheeks visually identical)."""
    circles = []
    if joint is not None:
        circles.append((joint[0], joint[1], 22.0))
    for c in (pulley, lobe):
        if c is not None:
            circles.append((c[0], c[1], HOUSING_R + 4.0))
    for px, pz in pivots:
        circles.append((px, pz, 9.0))
    if bridge:
        circles += rrect(*bridge, 6.0)
    y_in, y_out = sg * IH, sg * CO
    body = smooth_slab(circles, y_in, y_out, edge=1.6)
    if bridge:
        body = body.union(smooth_slab(rrect(*bridge, 6.0), sg * BH, y_in + sg * 0.6, edge=0))
    def tear(cx, cz, r, ya, yb):
        if teardrop:
            k = r * math.cos(math.radians(45))
            return poly_slab([(cx - k, cz + k), (cx + k, cz + k), (cx, cz + r * math.sqrt(2))], ya, yb)
        return None

    def hole(b, cx, cz, r, ya, yb):
        b = b.cut(circle_slab(cx, cz, r, ya, yb))
        t = tear(cx, cz, r, ya, yb)
        return b.cut(t) if t is not None else b

    if joint is not None:
        jx, jz = joint
        body = hole(body, jx, jz, SEAT_R, y_in, sg * (IH + 7))
        body = hole(body, jx, jz, 8.5, y_in, y_out)
        if enc:
            for dx in (-ENC["mount_holes"] / 2, ENC["mount_holes"] / 2):
                body = body.cut(circle_slab(jx + dx, jz, FIT["insert_m3_d"] / 2,
                                            y_out, sg * (CO - FIT["insert_m3_depth"])))
    for x, z in bridge_screws:
        body = body.cut(circle_slab(x, z, FIT["m3_clear"] / 2, sg * BH, y_out))
        body = body.cut(circle_slab(x, z, FIT["m3_head_d"] / 2, y_out, sg * (CO - FIT["m3_head_h"])))
    if pulley is not None:
        body = hole(body, pulley[0], pulley[1], TUBE_R + 0.5, y_in, y_out)
        for p in cap_screw_points(act_M):
            body = body.cut(circle_slab(p[0], p[2], FIT["m3_clear"] / 2, y_in, y_out))
            body = body.cut(circle_slab(p[0], p[2], FIT["m3_head_d"] / 2, y_in,
                                        sg * (IH + FIT["m3_head_h"])))
    for px, pz in pivots:
        if pivot_insert:
            body = body.cut(circle_slab(px, pz, 5.6 / 2, y_in, sg * (IH + 6.5)))
        else:
            body = body.cut(circle_slab(px, pz, FIT["m4_clear"] / 2, y_in, y_out))
            body = body.cut(circle_slab(px, pz, 4.0, y_out, sg * (CO - 1.0)))
    return Part(key, body, "printed", color, print_down=(0, sg, 0))


def cheek_hardware(sg, joint, act_M=None, bridge_screws=()):
    P = []
    jx, jz = joint
    P.append(Part("brg_608", hw.bearing("608"), "hardware", COLORS["bearing"],
                  M=frame_y(jx, jz, sg * IH, sg), bom="608 bearing (8x22x7)"))
    for x, z in bridge_screws:
        P.append(hwpart("shcs_m3x10", hw.shcs("M3", 10),
                        frame_y(x, z, sg * (CO - FIT["m3_head_h"]), sg), "M3x10 socket head"))
    if act_M is not None:
        for p in cap_screw_points(act_M):
            P.append(hwpart("shcs_m3x30", hw.shcs("M3", 30),
                            frame_y(p[0], p[2], sg * (IH + FIT["m3_head_h"]), -sg),
                            "M3x30 socket head (actuator through-bolt)"))
    return P


# ------------------------------------------------------------- bodies -------
def body_circles(bridge):
    return [(0.0, 0.0, HUB_R)] + rrect(*bridge, 6.0)


def link_body(key, s_prox, circles, bridge_screws, window=None, extra_circles=None, pad_from=None):
    """Link body (|y| <= BH): smooth tapered plank + hub + sector pad.
    pad_from: signed y where the pad starts (default: the far body face)."""
    b = smooth_slab(circles, -BH, BH, edge=2.5)
    if extra_circles:
        b = b.union(smooth_slab(extra_circles, -BH, BH, edge=2.5))
    y_lo, y_hi = ysort(-s_prox * BH if pad_from is None else pad_from, s_prox * SY0)
    pad = smooth_slab([(0, 0, PAD_R)], y_lo, y_hi, edge=1.2)
    b = b.union(pad)
    if window:
        x0, x1, h = window
        if x1 - x0 > 2 * h + 4:
            b = b.cut(smooth_slab([(x0 + h, 0, h), (x1 - h, 0, h)], -BH - 1, BH + 1, edge=0))
    b = b.cut(circle_slab(0, 0, SHAFT_BORE, -BH, BH))
    for a in sector_bolt_angles_for(key):
        x, z = pol(0, 0, 21.0, a)
        b = b.cut(circle_slab(x, z, FIT["insert_m3_d"] / 2, s_prox * SY0,
                              s_prox * SY0 - s_prox * FIT["insert_m3_depth"]))
    ys = s_prox * 10.5
    b = b.cut(cq.Workplane("XY").add(cq.Solid.makeCylinder(
        FIT["insert_m3_d"] / 2, FIT["insert_m3_depth"],
        cq.Vector(0, ys, HUB_R - FIT["insert_m3_depth"]), cq.Vector(0, 0, 1))))
    b = b.cut(cq.Workplane("XY").add(cq.Solid.makeCylinder(
        1.6, HUB_R, cq.Vector(0, ys, 0), cq.Vector(0, 0, 1))))
    for x, z in bridge_screws:
        for sg in (1, -1):
            b = b.cut(circle_slab(x, z, FIT["insert_m3_d"] / 2, sg * BH,
                                  sg * (BH - FIT["insert_m3_depth"])))
    return Part(key, b, "printed", C_W, print_down=(0, -s_prox, 0))


_BOLTS = {}


def _inside_any(pt, keepouts, margin):
    from scipy.spatial import Delaunay
    for circles in keepouts:
        pts = []
        for cx, cz, r in circles:
            for k in range(72):
                t = 2 * math.pi * k / 72
                pts.append((cx + (r + margin) * math.cos(t), cz + (r + margin) * math.sin(t)))
        if Delaunay(np.array(pts)).find_simplex(np.array([pt]))[0] >= 0:
            return True
    return False


def sector_bolt_angles(arc, keepouts=None):
    """Three bolt angles spread over the arc, clear of the link body."""
    a0, a1 = arc
    lo, hi = max(a0, 75.0) + 8, min(a1, 285.0) - 8
    cands = [a for a in np.linspace(lo, hi, 61)
             if not keepouts or not _inside_any(pol(0, 0, 21.0, a), keepouts, 4.5)]
    idx = np.linspace(0, len(cands) - 1, 3).round().astype(int)
    return [float(cands[i]) for i in idx]


def sector_bolt_angles_for(body_key):
    return _BOLTS[body_key]


def body_hardware(key, s_prox, bridge_screws):
    P = []
    ys = s_prox * 10.5
    P.append(hwpart("ins_m3", hw.heat_insert("M3"), T(0, ys, HUB_R), INSERT_BOM, COLORS["brass"]))
    P.append(hwpart("grub_m3x8", cq.Workplane("XY").circle(1.5).extrude(8),
                    T(0, ys, ARM["joint_shaft_d"] / 2), "M3x8 grub screw (shaft lock)"))
    for a in sector_bolt_angles_for(key):
        x, z = pol(0, 0, 21.0, a)
        P.append(hwpart("ins_m3", hw.heat_insert("M3"), frame_y(x, z, s_prox * SY0, s_prox),
                        INSERT_BOM, COLORS["brass"]))
        P.append(hwpart("shcs_m3x8", hw.shcs("M3", 8),
                        frame_y(x, z, s_prox * (SY1 - FIT["m3_head_h"]), s_prox),
                        "M3x8 socket head (sector)"))
    for x, z in bridge_screws:
        for sg in (1, -1):
            P.append(hwpart("ins_m3", hw.heat_insert("M3"), frame_y(x, z, sg * BH, sg),
                            INSERT_BOM, COLORS["brass"]))
    return P


def joint_shaft_parts(tag):
    P = [hwpart("shaft8_" + tag, hw.shaft(ARM["joint_shaft_d"], 2 * CO), frame_y(0, 0, -CO, 1),
                f"8 mm steel shaft x {2 * CO:g} mm (h6 / linear rod)", COLORS["steel"])]
    for sg in (1, -1):
        P.append(hwpart("shim_m8", hw.ring(14, 8.2, IH - BH), frame_y(0, 0, sg * BH, sg),
                        "M8 shim washer 8x14x0.5", COLORS["steel"]))
    return P


# ------------------------------------------------------------- sectors ------
def sector_features(Rc, arc):
    """Notch angle and clamp position for both ends of an arc."""
    a0, a1 = arc
    feats = []
    off = 5.0 if Rc > 30 else 3.0
    for a_end, sgn in ((a0, 1), (a1, -1)):
        an = a_end + sgn * deg_at(Rc, 2.5)
        rc = Rc - 7.0
        ac = an + sgn * deg_at(rc, off)
        feats.append(dict(notch=an, clamp=pol(0, 0, rc, ac), clamp_ang=ac, sgn=sgn))
    return feats


def notch_pts(Rc, a, w=1.6, depth=8.0):
    r0, r1 = Rc - depth, Rc + 2
    c, s = math.cos(math.radians(a)), math.sin(math.radians(a))
    tx, tz = -s, c
    return [(r0 * c - tx * w / 2, r0 * s - tz * w / 2), (r1 * c - tx * w / 2, r1 * s - tz * w / 2),
            (r1 * c + tx * w / 2, r1 * s + tz * w / 2), (r0 * c + tx * w / 2, r0 * s + tz * w / 2)]


def wedge_pts(Rc, arc, r_in):
    a0, a1 = arc
    r_root = Rc - CAP["cable_d"] / 2
    return arc_pts(0, 0, r_in, a0, a1) + arc_pts(0, 0, r_root, a1, a0)


def sector_windows(Rc, arc, r_in):
    """Curved spoke windows between r_in and the rim, clear of the clamps."""
    a0, a1 = arc
    r_out = Rc - 0.5 - 4.5
    w = r_out - r_in
    if w < 6:
        return []
    rm = (r_in + r_out) / 2
    lo = a0 + deg_at(rm, 16)
    hi = a1 - deg_at(rm, 16)
    span = hi - lo
    if span <= 0:
        return []
    spoke = deg_at(rm, 5.0)
    target = deg_at(rm, 26.0)
    n = max(1, round((span + spoke) / (target + spoke)))
    win = (span - (n - 1) * spoke) / n
    out = []
    for i in range(n):
        s0 = lo + i * (win + spoke)
        if win > deg_at(rm, w) * 1.05:
            out.append(kidney_pts(rm, w, s0 + deg_at(rm, w / 2), s0 + win - deg_at(rm, w / 2)))
    return out


def link_sector(key, s, Rc, arc, body_keepout, inner_r=HUB_R + 0.2):
    """Capstan sector in the link plane, slab y in s*[SY0, SY1]."""
    y_pad, y_out = s * SY0, s * SY1
    b = poly_slab(wedge_pts(Rc, arc, inner_r), y_pad, y_out)
    b = b.union(smooth_slab([(0, 0, PAD_R)], y_pad, y_out, edge=0))
    # clear the link body that runs through the slab
    lo, hi = ysort(y_pad, y_out)
    kos = body_keepout if isinstance(body_keepout[0][0], (list, tuple)) else [body_keepout]
    for ko in kos:
        b = b.cut(smooth_slab([(x, z, r + 0.8) for x, z, r in ko], lo - 1, hi + 1, edge=0))
    b = b.cut(circle_slab(0, 0, inner_r, y_pad, y_out))
    if body_keepout and not isinstance(body_keepout[0][0], (list, tuple)):
        body_keepout = [body_keepout]
    for a in sector_bolt_angles(arc, body_keepout):
        x, z = pol(0, 0, 21.0, a)
        b = b.cut(circle_slab(x, z, FIT["m3_clear"] / 2, y_pad, y_out))
        b = b.cut(circle_slab(x, z, FIT["m3_head_d"] / 2, y_out, y_out - s * FIT["m3_head_h"]))
    for f in sector_features(Rc, arc):
        b = b.cut(poly_slab(notch_pts(Rc, f["notch"]), y_pad, y_out))
        cx, cz = f["clamp"]
        b = b.cut(circle_slab(cx, cz, FIT["insert_m3_d"] / 2, y_out, y_out - s * FIT["insert_m3_depth"]))
    for pts in sector_windows(Rc, arc, PAD_R + 1.5):
        b = b.cut(poly_slab(pts, y_pad, y_out))
    return Part(key, b, "printed", C_A, print_down=(0, -s, 0))


_BOLTS_ARC = sector_bolt_angles


def clamp_hardware(M_list):
    P = []
    for M in M_list:
        P.append(hwpart("ins_m3", hw.heat_insert("M3"), M, INSERT_BOM, COLORS["brass"]))
        P.append(hwpart("washer_m3", hw.washer("M3"), M, "M3 washer (cable clamp)", COLORS["steel"]))
        P.append(hwpart("bhcs_m3x6", hw.bhcs("M3", 6), M @ T(0, 0, 0.5), "M3x6 button head (cable clamp)"))
    return P


def sector_hardware(s, Rc, arc, tag):
    y_out = s * SY1
    feats = sector_features(Rc, arc)
    P = clamp_hardware([frame_y(f["clamp"][0], f["clamp"][1], y_out, s) for f in feats])
    ymid = s * (SY0 + SY1) / 2
    P.append(Part("cable_" + tag, hw.cable_arc(Rc, feats[0]["notch"], feats[1]["notch"], 0.0,
                                               CAP["cable_d"]), "cable", COLORS["cable"],
                  M=frame((1, 0, 0), (0, 0, 1), (0, -1, 0), (0, ymid, 0))))
    return P


def cable_length(j):
    a0, a1 = j["arc"]
    rim = math.radians(a1 - a0) * j["Rc"]
    wrap = 2 * math.pi * CAP["pulley_r"] * CAP["turns"]
    return rim + wrap + 2 * 40


# ------------------------------------------------------- joint caps --------
CAP_H = 14.0
CAP_R = 22.0


def joint_cap():
    """Round cover over an output encoder.  Mount face z=0, outward +z.
    The encoder board hangs from two posts inside, chip facing the magnet."""
    wall, top = 1.6, 2.0
    m = ENC["mount_holes"] / 2
    c = cq.Workplane("XY").circle(CAP_R).extrude(CAP_H).faces(">Z").edges().fillet(4.0)
    c = c.cut(cq.Workplane("XY").circle(CAP_R - wall).extrude(CAP_H - top))
    c = c.union(cq.Workplane("XY").pushPoints([(-m, 0), (m, 0)]).circle(3.6).extrude(CAP_H - 1))
    c = c.intersect(cq.Workplane("XY").circle(CAP_R).extrude(CAP_H).faces(">Z").edges().fillet(4.0))
    board_top = 5.0 + ENC["board"][2]
    hs = ENC["board_holes"] / 2
    c = c.union(cq.Workplane("XY").workplane(offset=board_top).pushPoints([(0, -hs), (0, hs)])
                .circle(2.4).extrude(CAP_H - top - board_top + 0.5))
    c = c.cut(cq.Workplane("XY").workplane(offset=board_top).pushPoints([(0, -hs), (0, hs)])
              .circle(0.85).extrude(5))
    c = c.cut(cq.Workplane("XY").pushPoints([(-m, 0), (m, 0)]).circle(FIT["m3_clear"] / 2).extrude(CAP_H))
    c = c.cut(cq.Workplane("XY").workplane(offset=CAP_H - FIT["m3_head_h"])
              .pushPoints([(-m, 0), (m, 0)]).circle(FIT["m3_head_d"] / 2).extrude(FIT["m3_head_h"]))
    # wire exit + accent ring
    c = c.cut(cq.Workplane("XY").center(-CAP_R + 1, 0).rect(6, 9).extrude(4.0))
    c = c.cut(cq.Workplane("XY").workplane(offset=CAP_H - 0.6).circle(9.0).circle(8.0).extrude(1.0))
    return c


def encoder_set(M, tag, color=C_W):
    P = [Part("joint_cap", joint_cap(), "printed", color, M=M, print_down=(0, 0, 1))]
    P.append(hwpart("magnet", hw.magnet(), M, "Diametric magnet 6x2.5 mm", COLORS["magnet"]))
    P.append(Part("enc_board", hw.encoder_board(), "pcb", COLORS["pcb"], M=M @ T(0, 0, 5.0) @ R((0, 0, 1), 90),
                  bom="Encoder breakout (MT6701 / AS5047P), 24 mm M2 holes"))
    m = ENC["mount_holes"] / 2
    for dx in (-m, m):
        P.append(hwpart("ins_m3", hw.heat_insert("M3"), M @ T(dx, 0, 0), INSERT_BOM, COLORS["brass"]))
        P.append(hwpart("shcs_m3x16", hw.shcs("M3", 16), M @ T(dx, 0, CAP_H - FIT["m3_head_h"]),
                        "M3x16 socket head (joint cap)"))
    hs = ENC["board_holes"] / 2
    for dy in (-hs, hs):
        P.append(hwpart("screw_m2x6", hw.shcs("M2", 6), M @ T(0, dy, 5.0) @ R((1, 0, 0), 180),
                        "M2x6 self-tapping screw (encoder board)"))
    return P


# --------------------------------------------------------------- gears ------
def involute_gear_pts(m, z, phase_deg=0.0, n_flank=8):
    pa = math.radians(20)
    rp = m * z / 2
    rb = rp * math.cos(pa)
    ra = rp + m
    rf = rp - 1.25 * m
    inv = lambda a: math.tan(a) - a
    half_p = math.pi / (2 * z) + inv(pa)
    pts = []
    r_start = max(rb, rf)
    for k in range(z):
        c = 2 * math.pi * k / z + math.radians(phase_deg)
        flank = []
        for i in range(n_flank + 1):
            r = r_start + (ra - r_start) * i / n_flank
            a = math.acos(min(1.0, rb / r))
            flank.append((r, half_p - inv(a)))
        tooth = []
        if rf < rb:
            tooth.append((rf, c - half_p))
        tooth += [(r, c - th) for r, th in flank] + [(r, c + th) for r, th in reversed(flank)]
        if rf < rb:
            tooth.append((rf, c + half_p))
        pts += [(r * math.cos(a), r * math.sin(a)) for r, a in tooth]
    return pts


# ---------------------------------------------------------- arm geometry ----
J = JOINTS
Hb = ARM["base_h"]
TP = ARM["top_plate_t"]
BASE_R = ARM["base_size"] / 2 + 5.0
J1_OFF = (Hb - TP) - MF                    # actuator z -> world z for J1
_cp = STACK["cable_plane"] + J1_OFF
J1_DRUM_Z = (_cp - CAP["sector_t"] / 2, _cp + CAP["sector_t"] / 2)
TURRET_PLATE_Z0 = STACK["pul_front"][1] + J1_OFF + 0.5
TURRET_PLATE_Z1 = TURRET_PLATE_Z0 + 8.0
J2_Z = math.ceil(TURRET_PLATE_Z1 + J["J2"]["Rc"] + 4.0)
L1, L2 = ARM["L1"], ARM["L2"]
BOSS_Z0 = 30.0


def fork_geometry(L, jkey):
    j = J[jkey]
    d = j["d"]
    xb1 = L - d - TUBE_R - 3.0
    xb0 = xb1 - 20.0
    bridge = (xb0, -20.0, xb1, 20.0)
    screws = [(xb0 + 6, 13.0), (xb0 + 6, -13.0), (xb1 - 6, 0.0)]
    pulley = (L + d * math.cos(math.radians(j["dir"])), d * math.sin(math.radians(j["dir"])))
    return bridge, screws, pulley


# ----------------------------------------------------------------- base -----
def j1_act_matrix():
    return T(J["J1"]["d"], 0, J1_OFF)


def base_part():
    zt0 = Hb - TP
    ch = 10.0
    prof = [(0, 0), (BASE_R, 0), (BASE_R, Hb - ch), (BASE_R - ch, Hb), (0, Hb)]
    b = (cq.Workplane("XZ").polyline(prof).close().revolve(360, (0, 0, 0), (0, 1, 0)))
    inner = [(0, 0), (BASE_R - 4, 0), (BASE_R - 4, zt0 - ch + 4), (BASE_R - ch, zt0), (0, zt0)]
    b = b.cut(cq.Workplane("XZ").polyline(inner).close().revolve(360, (0, 0, 0), (0, 1, 0)))
    b = b.union(cq.Workplane("XY").workplane(offset=BOSS_Z0).circle(22.0).extrude(zt0 - BOSS_Z0))
    for ang in (90, 180, 270):
        rib = (cq.Workplane("XY").workplane(offset=BOSS_Z0)
               .rect(BASE_R - 6, 4).extrude(zt0 - BOSS_Z0)
               .translate(((BASE_R - 6) / 2, 0, 0)).rotate((0, 0, 0), (0, 0, 1), ang))
        b = b.union(rib)
    cols = [pol(0, 0, BASE_R - 12, a) for a in (45, 135, 225, 315)]
    b = b.union(cq.Workplane("XY").pushPoints(cols).circle(8).extrude(zt0))
    b = b.cut(cq.Workplane("XY").pushPoints(cols).circle(2.8).extrude(Hb))
    b = b.cut(cq.Workplane("XY").workplane(offset=Hb - 4).pushPoints(cols).circle(5).extrude(4))
    b = b.cut(cq.Workplane("XY").workplane(offset=Hb - 7).circle(SEAT_R).extrude(7))
    b = b.cut(cq.Workplane("XY").workplane(offset=BOSS_Z0).circle(SEAT_R).extrude(7))
    b = b.cut(cq.Workplane("XY").workplane(offset=BOSS_Z0).circle(8.5).extrude(Hb - BOSS_Z0))
    m = ENC["mount_holes"] / 2
    b = b.cut(cq.Workplane("XY").workplane(offset=BOSS_Z0).pushPoints([(-m, 0), (m, 0)])
              .circle(FIT["insert_m3_d"] / 2).extrude(FIT["insert_m3_depth"]))
    d1 = J["J1"]["d"]
    b = b.cut(cq.Workplane("XY").workplane(offset=zt0).center(d1, 0).circle(TUBE_R + 0.5).extrude(TP))
    for p in cap_screw_points(j1_act_matrix()):
        b = b.cut(cq.Workplane("XY").workplane(offset=zt0).center(p[0], p[1])
                  .circle(FIT["m3_clear"] / 2).extrude(TP))
        b = b.cut(cq.Workplane("XY").workplane(offset=Hb - FIT["m3_head_h"]).center(p[0], p[1])
                  .circle(FIT["m3_head_d"] / 2).extrude(FIT["m3_head_h"]))
    b = b.cut(cq.Workplane("XY").workplane(offset=zt0).center(-38, 0).slot2D(30, 12, 90).extrude(TP))
    # rear service port and a thin accent groove around the base
    b = b.cut(cq.Workplane("XY").box(14, 46, 22).translate((-BASE_R + 2, 0, 26)))
    groove = (cq.Workplane("XZ").polyline([(BASE_R - 0.8, 12), (BASE_R + 1, 12), (BASE_R + 1, 13.2),
                                           (BASE_R - 0.8, 13.2)]).close()
              .revolve(360, (0, 0, 0), (0, 1, 0)))
    b = b.cut(groove)
    return Part("base", b, "printed", C_G, print_down=(0, 0, 1))


TURRET_BOLTS = (30, 150, 210, 330)


def turret_drum():
    z0, z1 = J1_DRUM_Z
    Rc, arc = J["J1"]["Rc"], J["J1"]["arc"]
    hub_r = 30.0
    b = cq.Workplane("XY").workplane(offset=z0).circle(hub_r).extrude(TURRET_PLATE_Z0 - z0)
    try:
        b = b.faces(">Z").edges().chamfer(1.0)
    except Exception:
        pass
    b = b.union(cq.Workplane("XY").workplane(offset=z0).polyline(wedge_pts(Rc, arc, hub_r - 1))
                .close().extrude(z1 - z0))
    b = b.cut(cq.Workplane("XY").workplane(offset=z0).circle(SHAFT_BORE).extrude(TURRET_PLATE_Z0 - z0))
    for f in sector_features(Rc, arc):
        b = b.cut(cq.Workplane("XY").workplane(offset=z0).polyline(notch_pts(Rc, f["notch"])).close()
                  .extrude(z1 - z0))
        cx, cy = f["clamp"]
        b = b.cut(cq.Workplane("XY").workplane(offset=z1 - FIT["insert_m3_depth"]).center(cx, cy)
                  .circle(FIT["insert_m3_d"] / 2).extrude(FIT["insert_m3_depth"]))
    for pts in sector_windows(Rc, arc, hub_r + 1.5):
        b = b.cut(cq.Workplane("XY").workplane(offset=z0).polyline(pts).close().extrude(z1 - z0))
    for a in TURRET_BOLTS:
        x, y = pol(0, 0, 22.0, a)
        b = b.cut(cq.Workplane("XY").workplane(offset=TURRET_PLATE_Z0 - FIT["insert_m3_depth"])
                  .center(x, y).circle(FIT["insert_m3_d"] / 2).extrude(FIT["insert_m3_depth"]))
    zs = (z1 + TURRET_PLATE_Z0) / 2
    b = b.cut(cq.Workplane("YZ").workplane(offset=hub_r - FIT["insert_m3_depth"]).center(0, zs)
              .circle(FIT["insert_m3_d"] / 2).extrude(FIT["insert_m3_depth"] + 1))
    b = b.cut(cq.Workplane("YZ").center(0, zs).circle(1.6).extrude(hub_r))
    return Part("turret_drum", b, "printed", C_A, print_down=(0, 0, -1))


def turret_body():
    z0, z1 = TURRET_PLATE_Z0, TURRET_PLATE_Z1
    j = J["J2"]
    d = j["d"]
    px = d * math.cos(math.radians(j["dir"]))
    s = j["side"]
    M = act_matrix_link(px, J2_Z, s)
    base_rect = (-72.0, z0, 42.0, z1 + 10)
    plate = floor_slab([(0, 0, 58.0), (px, s * 18, 30.0), (px, -s * 18, 30.0)], z0, z1, edge=3.0,
                       top_only=True)
    cA = cheek("tmp", s, joint=(0, J2_Z), pulley=(px, J2_Z), act_M=M, bridge=base_rect, teardrop=True)
    cB = cheek("tmp", -s, joint=(0, J2_Z), lobe=(px, J2_Z), bridge=base_rect, enc=True, teardrop=True)
    b = plate.union(cA.shape).union(cB.shape)
    b = b.cut(cq.Workplane("XY").workplane(offset=z0).circle(5.0).extrude(z1 - z0))
    for a in TURRET_BOLTS:
        x, y = pol(0, 0, 22.0, a)
        b = b.cut(cq.Workplane("XY").workplane(offset=z0).center(x, y)
                  .circle(FIT["m3_clear"] / 2).extrude(z1 - z0))
        b = b.cut(cq.Workplane("XY").workplane(offset=z1 - FIT["m3_head_h"]).center(x, y)
                  .circle(FIT["m3_head_d"] / 2).extrude(FIT["m3_head_h"]))
    b = b.cut(cq.Workplane("XY").workplane(offset=z0).center(-38, 0).slot2D(24, 10, 90).extrude(z1 - z0))
    from . import balancer
    b = b.cut(balancer.turret_cuts())
    return Part("turret_body", b, "printed", C_W, print_down=(0, 0, -1)), M


# ---------------------------------------------------------------- links -----
def build_static():
    bodies = {k: [] for k in ("base", "turret", "upper", "fore")}

    # ---------- base + J1
    bodies["base"].append(base_part())
    Mj1 = j1_act_matrix()
    bodies["base"] += actuator_in(Mj1, "j1")
    for p in cap_screw_points(Mj1):
        bodies["base"].append(hwpart("shcs_m3x30", hw.shcs("M3", 30), T(p[0], p[1], Hb - FIT["m3_head_h"]),
                                     "M3x30 socket head (actuator through-bolt)"))
    for z in (Hb - 7, BOSS_Z0):
        bodies["base"].append(Part("brg_608", hw.bearing("608"), "hardware", COLORS["bearing"],
                                   M=T(0, 0, z), bom="608 bearing (8x22x7)"))
    bodies["base"] += encoder_set(T(0, 0, BOSS_Z0) @ R((1, 0, 0), 180), "j1", color=C_G)

    # ---------- turret
    tb, Mj2 = turret_body()
    bodies["turret"] += [turret_drum(), tb]
    z0, z1 = J1_DRUM_Z
    bodies["turret"].append(Part("j1_spacer", hw.ring(12, 8.3, z0 - Hb), "printed", C_G,
                                 M=T(0, 0, Hb), print_down=(0, 0, -1)))
    j1_len = TURRET_PLATE_Z0 - BOSS_Z0
    bodies["turret"].append(hwpart("shaft8_j1", hw.shaft(8, j1_len), T(0, 0, BOSS_Z0),
                                   f"8 mm steel shaft x {j1_len:.0f} mm", COLORS["steel"]))
    fts = sector_features(J["J1"]["Rc"], J["J1"]["arc"])
    bodies["turret"] += clamp_hardware([T(f["clamp"][0], f["clamp"][1], z1) for f in fts])
    bodies["turret"].append(Part("cable_j1", hw.cable_arc(J["J1"]["Rc"], fts[0]["notch"], fts[1]["notch"],
                                                          (z0 + z1) / 2, CAP["cable_d"]), "cable", COLORS["cable"]))
    for a in TURRET_BOLTS:
        x, y = pol(0, 0, 22.0, a)
        bodies["turret"].append(hwpart("shcs_m3x10", hw.shcs("M3", 10),
                                       T(x, y, TURRET_PLATE_Z1 - FIT["m3_head_h"]), "M3x10 socket head"))
        bodies["turret"].append(hwpart("ins_m3", hw.heat_insert("M3"), T(x, y, TURRET_PLATE_Z0),
                                       INSERT_BOM, COLORS["brass"]))
    bodies["turret"] += actuator_in(Mj2, "j2")
    s2 = J["J2"]["side"]
    bodies["turret"] += cheek_hardware(s2, (0, J2_Z), act_M=Mj2)
    bodies["turret"] += cheek_hardware(-s2, (0, J2_Z))
    bodies["turret"] += encoder_set(T(0, 0, J2_Z) @ frame_y(0, 0, -s2 * CO, -s2), "j2")
    from . import balancer
    bodies["turret"] += balancer.static_parts()

    # ---------- upper arm / forearm
    def link(name, L, s_prox, jkey_dist, sector_key, jkey_prox):
        P = []
        bridge, screws, pulley = fork_geometry(L, jkey_dist)
        jd, jp = J[jkey_dist], J[jkey_prox]
        circles = body_circles(bridge)
        _BOLTS[name] = sector_bolt_angles(jp["arc"], [circles])
        if name == "upper_arm":
            from . import balancer
            body = link_body(name, s_prox, circles, screws, window=(PAD_R + 10, bridge[0] - 8, 6.0))
            ear, cuts = balancer.ear_solids()
            body.shape = body.shape.union(ear).cut(cuts).cut(circle_slab(0, 0, SHAFT_BORE, -BH, BH))
            P.append(body)
            P += balancer.ear_pin_part()
        else:
            P.append(link_body(name, s_prox, circles, screws, window=(PAD_R + 10, bridge[0] - 8, 6.0)))
        P += body_hardware(name, s_prox, screws)
        P.append(link_sector(sector_key, s_prox, jp["Rc"], jp["arc"], circles))
        P += sector_hardware(s_prox, jp["Rc"], jp["arc"], jkey_prox.lower())
        P += joint_shaft_parts(jkey_prox.lower())
        sd = jd["side"]
        M = act_matrix_link(pulley[0], pulley[1], sd)
        P.append(cheek(name + "_cheek_a", sd, joint=(L, 0), pulley=pulley, act_M=M,
                       bridge=bridge, bridge_screws=screws))
        P.append(cheek(name + "_cheek_b", -sd, joint=(L, 0), lobe=pulley, bridge=bridge,
                       bridge_screws=screws, enc=True))
        P += cheek_hardware(sd, (L, 0), act_M=M, bridge_screws=screws)
        P += cheek_hardware(-sd, (L, 0), bridge_screws=screws)
        P += actuator_in(M, jkey_dist.lower())
        P += encoder_set(frame_y(L, 0, -sd * CO, -sd), jkey_dist.lower())
        return P

    bodies["upper"] = link("upper_arm", L1, J["J2"]["side"], "J3", "sector_j2", "J2")
    bodies["fore"] = link("forearm", L2, J["J3"]["side"], "J4", "sector_j3", "J3")

    # ---------- hand, wrist roll and gripper
    from . import hand
    bodies.update(hand.build())
    return bodies


# ------------------------------------------------------------ kinematics ----
def Ry_link(deg):
    return R((0, -1, 0), deg)


def body_frames(q):
    from . import hand
    Fb = np.eye(4)
    Ft = R((0, 0, 1), q["J1"])
    Fu = Ft @ T(0, 0, J2_Z) @ Ry_link(q["J2"])
    Ff = Fu @ T(L1, 0, 0) @ Ry_link(q["J3"])
    Fh = Ff @ T(L2, 0, 0) @ Ry_link(q["J4"])
    F = dict(base=Fb, turret=Ft, upper=Fu, fore=Ff, hand=Fh)
    F.update(hand.frames(Fh, q["J5"], q["G"]))
    return F


def assemble(bodies, q, dynamic=True):
    """Place every body for joint angles q.  dynamic=True adds the parts whose
    shape depends on the pose (balancer cable, piston, spring)."""
    F = body_frames(q)
    out = []
    for name, parts in bodies.items():
        for p in parts:
            pp = p.placed(F[name])
            pp.body = name
            out.append(pp)
    if dynamic:
        from . import balancer
        for p in balancer.dynamic_parts(q["J2"]):
            pp = p.placed(F["turret"])
            pp.body = "turret"
            out.append(pp)
    return out


HOME = dict(J1=0.0, J2=0.0, J3=0.0, J4=0.0, J5=0.0, G=10.0)
POSE = dict(J1=25.0, J2=55.0, J3=-85.0, J4=-30.0, J5=30.0, G=25.0)
