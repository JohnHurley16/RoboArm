"""Simple but dimensionally correct models of purchased hardware.

Every function returns a cq.Workplane in a local frame whose +Z is the part
axis.  Conventions are noted per function.
"""
import math
import cadquery as cq
from .params import ENC

BEARINGS = {  # name: (bore, OD, width)
    "608": (8, 22, 7),
    "6800": (10, 19, 5),
    "6801": (12, 21, 5),
    "6705": (25, 32, 4),
    "684": (4, 9, 4),
}

SCREW = {  # size: (head_d, head_h)
    "M2": (3.8, 2.0),
    "M3": (5.5, 3.0),
    "M4": (7.0, 4.0),
}

INSERT = {  # size: (OD, length)
    "M2": (3.2, 3.0),
    "M3": (4.6, 5.0),
}


def ring(od, id_, h, z0=0.0):
    return (cq.Workplane("XY").workplane(offset=z0)
            .circle(od / 2).circle(id_ / 2).extrude(h))


def bearing(name):
    """Bearing, z from 0 to width."""
    b, od, w = BEARINGS[name]
    t = (od - b) / 2
    outer = ring(od, od - t * 0.7, w)
    inner = ring(b + t * 0.7, b, w)
    seal = ring(od - t * 0.7, b + t * 0.7, w - 0.6, 0.3)
    return outer.union(inner).union(seal)


def rod(p0, p1, d):
    """Straight cable / rod between two 3D points."""
    a, b = cq.Vector(*p0), cq.Vector(*p1)
    v = b - a
    return cq.Workplane("XY").add(cq.Solid.makeCylinder(d / 2, v.Length, a, v.normalized()))


def dowel(d, length):
    """Dowel pin, z from 0 to length."""
    c = min(0.3, d * 0.1)
    return cq.Workplane("XY").circle(d / 2).extrude(length).edges().chamfer(c)


def shaft(d, length):
    c = 0.5
    return cq.Workplane("XY").circle(d / 2).extrude(length).edges().chamfer(c)


def shcs(size, length):
    """Socket head cap screw.  Head bearing face at z=0, head toward +z,
    shank toward -z (length = shank length)."""
    hd, hh = SCREW[size]
    d = float(size[1:])
    head = (cq.Workplane("XY").circle(hd / 2).extrude(hh)
            .faces(">Z").workplane().polygon(6, d * 0.9).cutBlind(-hh * 0.6))
    sh = cq.Workplane("XY").circle(d / 2).extrude(-length)
    return head.union(sh)


def bhcs(size, length):
    """Button head (ISO 7380).  Head bearing face at z=0, head toward +z."""
    d = float(size[1:])
    hd, hh = {"M3": (5.7, 1.65), "M4": (7.6, 2.2)}[size]
    head = (cq.Workplane("XY").circle(hd / 2).extrude(hh).faces(">Z").edges().fillet(hh * 0.6)
            .faces(">Z").workplane().polygon(6, d * 0.7).cutBlind(-hh * 0.5))
    return head.union(cq.Workplane("XY").circle(d / 2).extrude(-length))


def heat_insert(size):
    """Heat-set insert, z from 0 (flush face) down to -length."""
    od, ln = INSERT[size]
    d = float(size[1:])
    body = cq.Workplane("XY").circle(od / 2).extrude(-ln)
    body = body.faces(">Z").workplane().hole(d, ln)
    return body


def washer(size, t=0.5):
    d = float(size[1:])
    return ring(d * 2.1, d + 0.2, t)


def hex_nut(size):
    d = float(size[1:])
    af = {"M2": 4.0, "M3": 5.5, "M4": 7.0, "M5": 8.0}[size]
    h = {"M2": 1.6, "M3": 2.4, "M4": 3.2, "M5": 4.0}[size]
    return (cq.Workplane("XY").polygon(6, af / math.cos(math.pi / 6))
            .extrude(h).faces(">Z").workplane().hole(d))


def magnet():
    d, h = ENC["magnet"]
    return cq.Workplane("XY").circle(d / 2).extrude(h)


def encoder_board():
    """PCB with chip facing -z.  Board occupies z in [0, t], chip below 0."""
    w, l, t = ENC["board"]
    hs = ENC["board_holes"] / 2
    pcb = (cq.Workplane("XY").rect(w, l).extrude(t)
           .faces(">Z").workplane().pushPoints([(-hs, 0), (hs, 0)]).hole(2.2))
    chip = cq.Workplane("XY").rect(3.5, 3.5).extrude(-ENC["chip_t"])
    conn = (cq.Workplane("XY").workplane(offset=t)
            .center(0, l / 2 - 3).rect(10, 5).extrude(5))
    return pcb.union(chip).union(conn)


# ---------------------------------------------------------------- cable -----
def cable_arc(r, a0, a1, z, d):
    """Cable lying on a circle of radius r (XY plane) from angle a0..a1 (deg)."""
    if a1 < a0:
        a0, a1 = a1, a0
    path = cq.Workplane("XY").workplane(offset=z).add(
        cq.Edge.makeCircle(r, cq.Vector(0, 0, z), cq.Vector(0, 0, 1), a0, a1))
    start = cq.Vector(r * math.cos(math.radians(a0)),
                      r * math.sin(math.radians(a0)), z)
    tang = cq.Vector(-math.sin(math.radians(a0)), math.cos(math.radians(a0)), 0)
    prof = cq.Workplane(cq.Plane(origin=start, xDir=cq.Vector(0, 0, 1),
                                 normal=tang)).circle(d / 2)
    return prof.sweep(cq.Workplane().add(cq.Wire.assembleEdges(
        [cq.Edge.makeCircle(r, cq.Vector(0, 0, z), cq.Vector(0, 0, 1), a0, a1)])))


def cable_helix(r, pitch, turns, z0, d):
    h = pitch * turns
    hel = cq.Wire.makeHelix(pitch, h, r, center=cq.Vector(0, 0, z0))
    start = cq.Vector(r, 0, z0)
    tang = cq.Vector(0, 2 * math.pi * r, pitch).normalized()
    prof = cq.Workplane(cq.Plane(origin=start, xDir=cq.Vector(1, 0, 0),
                                 normal=tang)).circle(d / 2)
    return prof.sweep(cq.Workplane().add(hel), isFrenet=True)


def micro_servo(sv):
    """MG90S-class micro servo.  Output spline axis = +Z, spline top at z=0.
    Body long axis along X with the spline at x=0; body extends toward -x."""
    bx, bz, bh = sv["body"]
    off = sv["shaft_off"]
    cx = -(bx / 2 - off)                         # body centre x
    top = -(sv["spline_h"] + sv["boss_h"])       # body top face z
    body = (cq.Workplane("XY").workplane(offset=top - bh).center(cx, 0)
            .rect(bx, bz).extrude(bh))
    t0, t1 = sv["tab_h"]
    tabs = (cq.Workplane("XY").workplane(offset=top - bh + t0).center(cx, 0)
            .rect(sv["tab_len"], bz).extrude(t1 - t0))
    tabs = tabs.cut(cq.Workplane("XY").workplane(offset=top - bh + t0)
                    .pushPoints([(cx - sv["tab_holes"] / 2, 0), (cx + sv["tab_holes"] / 2, 0)])
                    .circle(1.25).extrude(t1 - t0))
    boss = cq.Workplane("XY").workplane(offset=top).circle(bz / 2 - 0.3).extrude(sv["boss_h"] - 1.0)
    boss = boss.union(cq.Workplane("XY").workplane(offset=top).center(-7.0, 0).circle(2.5)
                      .extrude(sv["boss_h"] - 2.0))
    spline = cq.Workplane("XY").workplane(offset=-sv["spline_h"]).circle(2.4).extrude(sv["spline_h"])
    lead = (cq.Workplane("XY").workplane(offset=top - bh + 3).center(cx - bx / 2, 0)
            .rect(4, 3).extrude(2))
    return dict(servo_body=body.union(tabs).union(boss).union(lead), servo_spline=spline)


def servo_horn():
    """Single-arm horn, hub on the spline.  z from 0 to 1.8, arm along +x."""
    h = cq.Workplane("XY").circle(3.6).extrude(1.8)
    arm = cq.Workplane("XY").polyline([(0, -2.6), (15, -1.8), (15, 1.8), (0, 2.6)]).close().extrude(1.8)
    return h.union(arm).cut(cq.Workplane("XY").circle(1.0).extrude(1.8))


def e_clip(d=8.0):
    return (cq.Workplane("XY").circle(d * 0.85).circle(d * 0.42).extrude(0.7)
            .cut(cq.Workplane("XY").center(d * 0.6, 0).rect(d * 0.7, d * 0.55).extrude(0.7)))


def nyloc(size):
    return hex_nut(size)


def gimbal_motor(m):
    """Hollow-shaft gimbal outrunner.  Stator base at z=0, rotor top at z=h.
    Returns named solids.  Rotor = bell + hollow shaft (rotates)."""
    r, h, bore = m["od"] / 2, m["h"], m["bore"]
    base_h = 3.0
    base = (cq.Workplane("XY").circle(r - 1.5).circle(bore / 2 + 1.6).extrude(base_h))
    n, sd, bc = m["stator_holes"]
    base = base.cut(cq.Workplane("XY").polarArray(bc / 2, 45, 360, n).circle(sd / 2 * 0.85).extrude(base_h))
    bell = (cq.Workplane("XY").workplane(offset=base_h + 0.6).circle(r).circle(bore / 2 + 0.01).extrude(h - base_h - 0.6)
            .faces(">Z").edges().fillet(1.2))
    n, sd, bc = m["rotor_holes"]
    bell = bell.cut(cq.Workplane("XY").workplane(offset=h - 3).polarArray(bc / 2, 45, 360, n)
                    .circle(sd / 2 * 0.85).extrude(3))
    groove = cq.Workplane("XY").workplane(offset=h * 0.55).circle(r + 1).circle(r - 0.4).extrude(0.8)
    bell = bell.cut(groove)
    shaft = cq.Workplane("XY").circle(bore / 2 + 1.5).circle(bore / 2).extrude(h)
    stator = (cq.Workplane("XY").workplane(offset=base_h + 1.5).circle(r - 3.5).circle(bore / 2 + 3).extrude(h - base_h - 4))
    lead = cq.Workplane("XY").add(cq.Solid.makeCylinder(1.6, 7, cq.Vector(0, r - 3.5, 1.5), cq.Vector(0, 0, -1)))
    return dict(motor_base=base, motor_bell=bell, motor_shaft=shaft, motor_stator=stator, motor_wires=lead)


def moteus_c1(d):
    """moteus-c1 class driver: board z 0..1.6, components on +z side,
    encoder chip at the centre of the -z side (faces the motor)."""
    w, l, t = d["board"]
    pcb = cq.Workplane("XY").rect(w, l).extrude(1.6).edges("|Z").fillet(2)
    hs = d["holes"][0] / 2
    pcb = pcb.cut(cq.Workplane("XY").pushPoints([(hs, hs), (-hs, hs), (-hs, -hs), (hs, -hs)]).circle(1.1).extrude(1.6))
    parts = cq.Workplane("XY").workplane(offset=1.6).center(-6, 4).rect(16, 14).extrude(2.5)
    xt30 = cq.Workplane("XY").workplane(offset=1.6).center(6, -12).rect(10, 5.5).extrude(t - 1.6)
    ph = cq.Workplane("XY").workplane(offset=1.6).center(-5, -13.5).rect(8, 4.5).extrude(4.5)
    cap = cq.Workplane("XY").workplane(offset=1.6).center(8, 9).circle(3).extrude(6)
    chip = cq.Workplane("XY").rect(3.5, 3.5).extrude(-0.9)
    return dict(drv_pcb=pcb, drv_parts=parts.union(xt30).union(ph).union(cap), drv_chip=chip)
