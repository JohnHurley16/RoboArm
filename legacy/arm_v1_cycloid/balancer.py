"""Shoulder (J2) gravity balancer: zero-length cable spring.

Geometry (turret link plane, J2 axis = O at (0, J2_Z)):
  * an ear on the upper arm carries a 3 mm pin at child angle BETA, radius B_R
  * the cable runs from that pin to an idler pulley on the turret at angle
    ALPHA, radius A_R (ALPHA - BETA = 90 deg  ->  spring torque ~ cos(q2))
  * it leaves the idler along its underside, runs back into a spring
    cartridge behind the turret, round a 2:1 pulley on the piston and back
    to a tension screw.  The spring is compressed, so it fails safe.
Cable tension T = k_eff * (L - L0): with the tension screw set right this is
a zero-length spring and the torque follows the arm's gravity torque.
"""
import math
import numpy as np
import cadquery as cq

from .params import FIT, ARM, JOINTS
from . import hardware as hw
from .core import Part, COLORS, T, R, frame, smooth_slab, hull_sketch

ALPHA, A_R = 220.0, 66.0              # idler position (deg, mm) from J2
BETA = ALPHA - 90.0                   # ear direction in the upper-arm frame
B_HOLES = (44.0, 47.0, 50.0)          # tuning holes in the ear (nominal = middle)
B_R = B_HOLES[1]
R_IDL = 6.5                           # cable-centre radius on the idler (V-groove root 6)
PLANE_Y = -8.0                        # cable plane (pad side of the upper arm)
IDL_Y = (-10.5, -5.5)
R_PP = 5.0                            # piston pulley cable radius (2:1)
TUBE_ID, TUBE_OD = 23.0, 28.0
BULK_X = (-102.0, -94.0)              # bulkhead (front cap of the cartridge)
FL_R, FL_HOLE_R = 21.0, 17.0          # tube flange / bulkhead radius, flange screw circle
EAR_Y = (-13.5, -9.5)                 # ear plate (prints on the bed with the upper arm)

C_W, C_G, C_A = COLORS["printed"], COLORS["printed2"], COLORS["accent"]


def J2Z():
    from .arm import J2_Z
    return J2_Z


def idler_xz():
    a = math.radians(ALPHA)
    return (A_R * math.cos(a), J2Z() + A_R * math.sin(a))


def pod_axis_z():
    return idler_xz()[1] - R_IDL - R_PP          # incoming strand on top of the piston pulley


def strand_z():
    return idler_xz()[1] - R_IDL                 # cable height from idler bottom into the pod


# --------------------------------------------------------------- physics ----
def pin_world(q2, b=B_R):
    th = math.radians(BETA + q2)
    return np.array([b * math.cos(th), J2Z() + b * math.sin(th)])


def tangent(q2, b=B_R):
    """Free segment from the ear pin to the idler (cable arrives on the side
    facing the shoulder and wraps clockwise to leave along the underside)."""
    P = pin_world(q2, b)
    A = np.array(idler_xz())
    best = None
    for phi in np.linspace(-89.9, 269.9, 3601):
        p = math.radians(phi)
        Tp = A + R_IDL * np.array([math.cos(p), math.sin(p)])
        d = Tp - P
        n = np.linalg.norm(d)
        if n < 1e-6:
            continue
        cw = np.array([math.sin(p), -math.cos(p)])
        err = 1 - np.dot(d / n, cw)
        if best is None or err < best[0]:
            best = (err, phi, Tp, n)
    _, phi, Tp, n = best
    wrap = math.radians(phi + 90.0)
    return P, Tp, n, wrap


def path_length(q2, b=B_R):
    _, _, n, wrap = tangent(q2, b)
    return n + R_IDL * wrap


def spring_torque(q2, k_eff, L0, b=B_R):
    """Torque on the upper arm (N m, + = lifting) for cable rate k_eff (N/mm)."""
    P, Tp, n, _ = tangent(q2, b)
    Tn = max(0.0, k_eff * (path_length(q2, b) - L0))
    u = (Tp - P) / n
    r = (P - np.array([0.0, J2Z()])) / 1000.0
    return (r[0] * u[1] - r[1] * u[0]) * Tn, Tn


# ----------------------------------------------------------------- parts ----
def ear_solids():
    """Ear plate for the upper-arm body (child frame) + its pin holes."""
    ux, uz = math.cos(math.radians(BETA)), math.sin(math.radians(BETA))
    circles = [(0.0, 0.0, 15.0)] + [(b * ux, b * uz, 4.5) for b in (B_HOLES[0], B_HOLES[-1])]
    ear = smooth_slab(circles, *EAR_Y, edge=0.8)
    cuts = None
    for b in B_HOLES:
        c = _cyl_y(b * ux, b * uz, 1.5 - FIT["press"], *EAR_Y)
        cuts = c if cuts is None else cuts.union(c)
    return ear, cuts


def spool():
    """Printed spool on the ear pin: keeps the cable loop in its plane."""
    prof = [(1.6, 0), (4.5, 0), (4.5, 0.6), (2.6, 1.0), (2.6, 2.0), (4.5, 2.4), (4.5, 3.0), (1.6, 3.0)]
    return cq.Workplane("XZ").polyline(prof).close().revolve(360, (0, 0, 0), (0, 1, 0))


def _cyl_y(x, z, r, y0, y1):
    from .core import cyl_y
    return cyl_y(x, z, r, min(y0, y1), max(y0, y1))


def ear_pin_part():
    ux, uz = math.cos(math.radians(BETA)), math.sin(math.radians(BETA))
    from .arm import frame_y
    x, z = B_R * ux, B_R * uz
    return [Part("dowel_3x8", hw.dowel(3, 8), "hardware", COLORS["steel"],
                 M=frame_y(x, z, EAR_Y[0], 1), bom="Dowel pin 3x8 (balancer ear)"),
            Part("balancer_spool", spool(), "printed", C_A, M=frame_y(x, z, PLANE_Y - 1.5, 1),
                 print_down=(0, 0, -1))]


def idler():
    """V-groove idler (45 deg walls print without support).  Axis = local Z, z 0..w."""
    y0, y1 = IDL_Y
    w = y1 - y0
    root = R_IDL - 0.75                 # 1 mm cable in a 90 deg V sits ~0.7 above the root
    h = (w - 0.6) / 2
    fl = root + h                       # 45 deg groove walls
    prof = [(2.3, 0), (fl, 0), (fl, 0.3), (root, 0.3 + h), (fl, w - 0.3), (fl, w), (2.3, w)]
    pul = cq.Workplane("XZ").polyline(prof).close().revolve(360, (0, 0, 0), (0, 1, 0))
    pul = pul.cut(cq.Workplane("XY").circle(4.5 + FIT["bearing_seat"]).extrude(4.0).translate((0, 0, (w - 4) / 2)))
    return pul


def turret_cuts():
    """Holes the turret body needs (turret frame)."""
    from .arm import frame_y, circle_slab, IH, CO
    ax, az = idler_xz()
    c = circle_slab(ax, az, 2.2, -CO, -IH)
    c = c.union(circle_slab(ax, az, 4.0, -CO, -CO + 3.0))           # counterbore for M4 head
    c = c.union(circle_slab(ax, az, 5.5, -IH - 0.8, -IH + 0.6))     # clear the cheek's inner bridge tab
    for x in FOOT_BOLTS_X:
        c = c.union(cq.Workplane("XY").workplane(offset=_plate_top() - FIT["insert_m3_depth"])
                    .center(x, PLANE_Y).circle(FIT["insert_m3_d"] / 2).extrude(FIT["insert_m3_depth"]))
    return c


FOOT_X = (-84.0, -62.0)
FOOT_BOLTS_X = (-78.0, -68.0)


def _plate_top():
    from .arm import TURRET_PLATE_Z1
    return TURRET_PLATE_Z1


def bracket():
    """Foot on the turret plate + bulkhead (front cap of the cartridge)."""
    zt = _plate_top()
    zc = pod_axis_z()
    bx0, bx1 = BULK_X
    foot_h = 5.0
    b = cq.Workplane("XY").add(cq.Solid.makeBox(FOOT_X[1] - (bx1 - 1.0), FOOT_Y[1] - FOOT_Y[0], foot_h,
                                                cq.Vector(bx1 - 1.0, FOOT_Y[0], zt)))
    try:
        b = b.edges("|X").fillet(1.0)
    except Exception:
        pass
    bulk = cq.Workplane("XY").add(cq.Solid.makeCylinder(FL_R, bx1 - bx0, cq.Vector(bx0, PLANE_Y, zc),
                                                        cq.Vector(1, 0, 0)))
    try:
        bulk = bulk.faces(">X").edges().chamfer(1.0)
    except Exception:
        pass
    b = b.union(bulk)
    for x in FOOT_BOLTS_X:
        b = b.cut(cq.Workplane("XY").add(cq.Solid.makeCylinder(FIT["m3_clear"] / 2, foot_h, cq.Vector(x, PLANE_Y, zt),
                                                               cq.Vector(0, 0, 1))))
        b = b.cut(cq.Workplane("XY").add(cq.Solid.makeCylinder(3.3, 2.0, cq.Vector(x, PLANE_Y, zt + foot_h - 2.0),
                                                               cq.Vector(0, 0, 1))))
    b = b.cut(cq.Workplane("XY").add(cq.Solid.makeCylinder(1.5, 20, cq.Vector(bx0 - 5, PLANE_Y, strand_z()),
                                                           cq.Vector(1, 0, 0))))
    b = b.cut(cq.Workplane("XY").add(cq.Solid.makeCylinder(1.5, 20, cq.Vector(bx0 - 5, PLANE_Y, anchor_z()),
                                                           cq.Vector(1, 0, 0))))
    # cable clamp insert on the front face, beside the anchor hole
    cy, cz = _clamp_yz()
    b = b.cut(cq.Workplane("XY").add(cq.Solid.makeCylinder(FIT["insert_m3_d"] / 2, FIT["insert_m3_depth"],
                                                           cq.Vector(bx1 - FIT["insert_m3_depth"], cy, cz),
                                                           cq.Vector(1, 0, 0))))
    for y, z in flange_holes():
        b = b.cut(cq.Workplane("XY").add(cq.Solid.makeCylinder(FIT["insert_m3_d"] / 2, FIT["insert_m3_depth"],
                                                               cq.Vector(bx0, y, z), cq.Vector(1, 0, 0))))
    return Part("balancer_bracket", b, "printed", C_G, print_down=(-1, 0, 0))


FOOT_Y = (-13.0, -3.0)


def anchor_z():
    return strand_z() - 2 * R_PP


def flange_holes():
    zc = pod_axis_z()
    return [(PLANE_Y + FL_HOLE_R * math.cos(math.radians(a)), zc + FL_HOLE_R * math.sin(math.radians(a)))
            for a in (90, 210, 330)]


FLANGE_T = 3.0


def tube_len():
    f = fitted()
    longest = f["spring"]["free"] - f["defl_min"]
    return math.ceil(longest + PISTON_L + 3.0 + 2.4)


def tube():
    """Cartridge tube: closed rear end, flange at the front with a 45 deg
    underside (prints closed-end down without support)."""
    bx0 = BULK_X[0]
    zc = pod_axis_z()
    L = tube_len()
    ro, ri = TUBE_OD / 2, TUBE_ID / 2
    xf = L - FLANGE_T            # local x: 0 = closed end, L = flange face
    prof = [(0, 0), (ro - 4, 0), (ro, 4), (ro, xf - (FL_R - ro)), (FL_R, xf), (FL_R, L), (ri, L), (ri, 2.4), (0, 2.4)]
    t = (cq.Workplane("XY").polyline([(x, r) for r, x in prof]).close()
         .revolve(360, (0, 0, 0), (1, 0, 0)))
    for y, z in flange_holes():
        t = t.cut(cq.Workplane("XY").add(cq.Solid.makeCylinder(FIT["m3_clear"] / 2, FLANGE_T + 8,
                                                               cq.Vector(xf - 8, y - PLANE_Y, z - zc), cq.Vector(1, 0, 0))))
        t = t.cut(cq.Workplane("XY").add(cq.Solid.makeCylinder(3.3, 8, cq.Vector(xf - 8, y - PLANE_Y, z - zc),
                                                               cq.Vector(1, 0, 0))))   # head pocket in the cone
    t = t.cut(cq.Workplane("XY").add(cq.Solid.makeBox(L - 34, 6, 30, cq.Vector(14, 9.5, -15))))   # sight slot
    t = t.translate((bx0 - L, PLANE_Y, zc))
    return Part("balancer_tube", t, "printed", C_W, print_down=(-1, 0, 0))


PISTON_L = 16.0
PP_X = -9.5       # piston pulley pin position (piston frame)


def piston():
    """Piston in its own frame: front face at x=0, extends to -PISTON_L.
    Holds the 2:1 pulley on an M3 pin along y at x = -6."""
    p = cq.Workplane("XY").add(cq.Solid.makeCylinder(TUBE_ID / 2 - 0.3, 3.0, cq.Vector(-3.0, 0, 0), cq.Vector(1, 0, 0)))
    for ys in (-4.5, 3.0):
        p = p.union(cq.Workplane("XY").add(cq.Solid.makeBox(PISTON_L - 3.0, 1.5, 18.0,
                                                            cq.Vector(-PISTON_L, ys, -9.0))))
    p = p.union(cq.Workplane("XY").add(cq.Solid.makeBox(PISTON_L - 3.0, 9.0, 3.0, cq.Vector(-PISTON_L, -4.5, -11.2)))
                .intersect(cq.Workplane("XY").add(cq.Solid.makeCylinder(TUBE_ID / 2 - 0.3, PISTON_L,
                                                                        cq.Vector(-PISTON_L, 0, 0), cq.Vector(1, 0, 0)))))
    # cable holes through the face (two strands) + pin hole
    for z in (R_PP, -R_PP):
        p = p.cut(cq.Workplane("XY").add(cq.Solid.makeCylinder(1.4, 4.0, cq.Vector(-3.5, 0, z), cq.Vector(1, 0, 0))))
    p = p.cut(cq.Workplane("XY").add(cq.Solid.makeCylinder(1.5, 12.0, cq.Vector(PP_X, -6.0, 0), cq.Vector(0, 1, 0))))
    return p


PP_W = 3.4


def piston_pulley():
    root = R_PP - 0.75
    prof = [(1.6, 0), (root + 1.4, 0), (root, 1.4), (root, PP_W - 1.4), (root + 1.4, PP_W), (1.6, PP_W)]
    return cq.Workplane("XZ").polyline(prof).close().revolve(360, (0, 0, 0), (0, 1, 0))


def spring_solid(length, od, wire, coils):
    D = od - wire
    pitch = (length - wire) / coils
    hel = cq.Wire.makeHelix(pitch, length - wire, D / 2, center=cq.Vector(0, 0, wire / 2))
    prof = cq.Workplane(cq.Plane(origin=(D / 2, 0, wire / 2), xDir=(1, 0, 0),
                                 normal=cq.Vector(0, math.pi * D, pitch).normalized())).circle(wire / 2)
    return prof.sweep(cq.Workplane().add(hel), isFrenet=True)


# --------------------------------------------------------- spring sizing ----
G_STEEL = 79000.0       # N/mm^2


def spring_design(k_s, defl_max, f_max):
    """Pick a stock-looking music-wire compression spring for the cartridge."""
    best = None
    for wire in (1.8, 2.0, 2.2, 2.5, 2.8, 3.0):
        for od in (19.0, 20.0, 21.0):
            D = od - wire
            na = G_STEEL * wire ** 4 / (8 * D ** 3 * k_s)
            if na < 3:
                continue
            nt = na + 2
            solid = nt * wire
            free = defl_max + solid * 1.15 + 2
            C = D / wire
            kw = (4 * C - 1) / (4 * C - 4) + 0.615 / C
            tau = kw * 8 * f_max * D / (math.pi * wire ** 3)
            sut = 2211 * wire ** -0.145          # ASTM A228 music wire (MPa)
            if tau > 0.40 * sut:
                continue
            score = free
            if best is None or score < best["free"]:
                best = dict(wire=wire, od=od, D=D, active=na, total=nt, solid=solid, free=free, tau=tau,
                            allow=0.40 * sut, rate=k_s)
    return best


# ------------------------------------------------- assembly (turret frame) --
import json as _json
import os as _os

_FIT = None


def fitted():
    """Cable rate / preload / spring chosen by analysis_balance.py (cached)."""
    global _FIT
    if _FIT is None:
        path = _os.path.join(_os.path.dirname(__file__), "balancer_fit.json")
        _FIT = _json.load(open(path))
    return _FIT


def _xz_frame(origin_xyz):
    """Local XY plane -> link x/z plane at the given point (local Z -> -y)."""
    return frame((1, 0, 0), (0, 0, 1), (0, -1, 0), origin_xyz)


def _ax_frame(origin, zdir):
    zdir = np.asarray(zdir, float) / np.linalg.norm(zdir)
    if abs(zdir[0]) > 0.9:
        x = np.cross((0, 0, 1.0), zdir)
    else:
        x = np.cross((1.0, 0, 0), zdir)
    x /= np.linalg.norm(x)
    y = np.cross(zdir, x)
    return frame(x, y, zdir, origin)


def static_parts():
    from .arm import frame_y, IH, CO, hwpart, INSERT_BOM
    P = []
    ax, az = idler_xz()
    # idler on a cantilevered M4 bolt from the encoder-side cheek
    P.append(Part("balancer_idler", idler(), "printed", C_A, M=frame_y(ax, az, IDL_Y[0], 1), print_down=(0, 0, -1)))
    P.append(Part("brg_684", hw.bearing("684"), "hardware", COLORS["bearing"],
                  M=frame_y(ax, az, (IDL_Y[0] + IDL_Y[1]) / 2 - 2.0, 1), bom="684ZZ bearing (4x9x4)"))
    P.append(Part("balancer_spacer", hw.ring(8.0, 4.3, IDL_Y[0] + IH), "printed", C_G,
                  M=frame_y(ax, az, -IH, 1), print_down=(0, 0, -1)))
    P.append(hwpart("shcs_m4x20", hw.shcs("M4", 20), frame_y(ax, az, -CO + 3.0, -1), "M4x20 socket head (balancer idler)"))
    P.append(hwpart("nyloc_m4", hw.hex_nut("M4"), frame_y(ax, az, IDL_Y[1], 1), "M4 nyloc nut"))
    # bracket + cartridge
    P.append(bracket())
    zt = _plate_top()
    for x in FOOT_BOLTS_X:
        P.append(hwpart("ins_m3", hw.heat_insert("M3"), T(x, PLANE_Y, zt), INSERT_BOM, COLORS["brass"]))
        P.append(hwpart("bhcs_m3x8", hw.bhcs("M3", 8), T(x, PLANE_Y, zt + 3.0), "M3x8 button head"))
    P.append(tube())
    zc = pod_axis_z()
    for y, z in flange_holes():
        P.append(hwpart("ins_m3", hw.heat_insert("M3"), _ax_frame((BULK_X[0], y, z), (1, 0, 0)), INSERT_BOM,
                        COLORS["brass"]))
        P.append(hwpart("bhcs_m3x8", hw.bhcs("M3", 8), _ax_frame((BULK_X[0] - FLANGE_T, y, z), (-1, 0, 0)),
                        "M3x8 button head"))
    # anchor strand: clamped under a washer on the bulkhead front face
    cy, cz = _clamp_yz()
    P.append(hwpart("ins_m3", hw.heat_insert("M3"), _ax_frame((BULK_X[1], cy, cz), (1, 0, 0)), INSERT_BOM,
                    COLORS["brass"]))
    P.append(hwpart("washer_m3", hw.washer("M3"), _ax_frame((BULK_X[1], cy, cz), (1, 0, 0)), "M3 washer (cable clamp)",
                    COLORS["steel"]))
    P.append(hwpart("bhcs_m3x6", hw.bhcs("M3", 6), _ax_frame((BULK_X[1] + 0.5, cy, cz), (1, 0, 0)),
                    "M3x6 button head (cable clamp)"))
    # fixed cable run: idler underside -> bulkhead
    P.append(Part("cable_bal_run", hw.rod((ax, PLANE_Y, strand_z()), (BULK_X[1], PLANE_Y, strand_z()), 1.0),
                  "cable", COLORS["cable"]))
    return P


def spring_length(q2):
    f = fitted()
    L = path_length(q2)
    return f["spring"]["free"] - (L - f["L0"]) / 2


_PISTON = {}


def dynamic_parts(q2):
    """Pose-dependent parts in the turret frame."""
    f = fitted()
    sp = f["spring"]
    P = []
    Pp, Tp, n, wrap = tangent(q2)
    P.append(Part("cable_bal_free", hw.rod((Pp[0], PLANE_Y, Pp[1]), (Tp[0], PLANE_Y, Tp[1]), 1.0),
                  "cable", COLORS["cable"]))
    ax, az = idler_xz()
    phi = math.degrees(wrap) - 90.0
    if wrap > 0.05:
        P.append(Part("cable_bal_wrap", hw.cable_arc(R_IDL, -90.0, phi, 0.0, 1.0), "cable", COLORS["cable"],
                      M=_xz_frame((ax, PLANE_Y, az))))
    s = spring_length(q2)
    x_face = BULK_X[0] - s
    zc = pod_axis_z()
    if "p" not in _PISTON:
        _PISTON["p"] = (piston(), piston_pulley())
    pis, pul = _PISTON["p"]
    Mp = T(x_face, PLANE_Y, zc)
    P.append(Part("balancer_piston", pis, "printed", C_A, M=Mp, print_down=(1, 0, 0)))
    P.append(Part("balancer_piston_pulley", pul, "printed", C_G,
                  M=Mp @ frame((1, 0, 0), (0, 0, -1), (0, 1, 0), (PP_X, -PP_W / 2, 0)), print_down=(0, 0, -1)))
    P.append(Part("dowel_3x12", hw.dowel(3, 12), "hardware", COLORS["steel"],
                  M=Mp @ frame((1, 0, 0), (0, 0, -1), (0, 1, 0), (PP_X, -6.0, 0)), bom="Dowel pin 3x12 (piston pulley)"))
    P.append(Part("balancer_spring", spring_solid(s, sp["od"], sp["wire"], sp["total"] - 0.5), "hardware",
                  (0.62, 0.64, 0.68), M=_ax_frame((BULK_X[0], PLANE_Y, zc), (-1, 0, 0)),
                  bom=f"Compression spring, music wire {sp['wire']:g} mm, OD {sp['od']:g} mm, "
                      f"free length {math.ceil(sp['free']):d} mm, ~{sp['rate']:.1f} N/mm, ends closed & ground"))
    # strands inside the cartridge
    for z in (strand_z(), strand_z() - 2 * R_PP):
        P.append(Part("cable_bal_inner", hw.rod((BULK_X[1], PLANE_Y, z), (x_face + PP_X, PLANE_Y, z), 1.0),
                      "cable", COLORS["cable"]))
    return P


def _clamp_yz():
    return (PLANE_Y, anchor_z() - 6.0)

