"""Direct capstan actuator: gimbal motor -> zero-backlash jaw coupling ->
capstan pulley on a preloaded pair of 6705 bearings.  No gearbox.

Actuator frame: +Z is the motor axis pointing into the joint, Z = 0 is the
mounting face (the fork cheek's outer face).  The motor, driver and rear lid
sit at Z < 0, the pulley at Z > 0.
"""
import math
import numpy as np
import cadquery as cq

from .params import (MOTORS, DRIVER, FIT, CAP, STACK as S, HOUSING_R, CAP_SCREW_R, TUBE_R, motor_stack)
from . import hardware as hw
from .core import Part, COLORS, T, R

BRG_SEAT = lambda name: hw.BEARINGS[name][1] / 2 + FIT["bearing_seat"]
CAP_SCREW_ANGLES = [45, 135, 225, 315]
CAP_BORE = 19.0
PLATE_SCREWS = [30, 150, 270]     # radial screws holding the rear plate
LID_SCREWS = [90, 210, 330]       # radial screws holding the rear lid
JAW_R = (7.0, 14.0)
DOG_DEG, LEG_DEG = 38.0, 22.0


def revolve(pts):
    return (cq.Workplane("XZ").polyline(pts).close().revolve(360, (0, 0, 0), (0, 1, 0)))


def polar(r, angles):
    return [(r * math.cos(math.radians(a)), r * math.sin(math.radians(a))) for a in angles]


def sector_ring(r0, r1, a0, a1, z0, z1, n=24):
    pts = [(r1 * math.cos(math.radians(a0 + (a1 - a0) * i / n)), r1 * math.sin(math.radians(a0 + (a1 - a0) * i / n)))
           for i in range(n + 1)]
    pts += [(r0 * math.cos(math.radians(a1 - (a1 - a0) * i / n)), r0 * math.sin(math.radians(a1 - (a1 - a0) * i / n)))
            for i in range(n + 1)]
    return cq.Workplane("XY").workplane(offset=z0).polyline(pts).close().extrude(z1 - z0)


def dogs(start_deg, z0, z1, width=DOG_DEG):
    out = None
    for k in range(3):
        c = start_deg + 120 * k
        d = sector_ring(JAW_R[0], JAW_R[1], c - width / 2, c + width / 2, z0, z1)
        out = d if out is None else out.union(d)
    return out


# ------------------------------------------------------------ front cap -----
def front_cap():
    """Bearing nose: flange outside the cheek + tube through it."""
    z0, z1 = S["cap_ring"]
    s1 = BRG_SEAT("6705")
    pts = [(CAP_BORE, z0), (HOUSING_R, z0), (HOUSING_R, z1), (TUBE_R, z1),
           (TUBE_R, S["tube_end"]), (s1, S["tube_end"]), (s1, S["lip"][1]),
           (14.0, S["lip"][1]), (14.0, S["lip"][0] + (s1 - 14.0)), (s1, S["lip"][0]),
           (s1, z1), (CAP_BORE, z1)]
    cap = revolve(pts)
    try:
        cap = cap.faces("<Z").edges(cq.selectors.RadiusNthSelector(-1)).chamfer(0.8)
    except Exception:
        pass
    # inserts for the cheek screws (from the mounting face) + clearance for the can screws
    cap = cap.cut(cq.Workplane("XY").workplane(offset=z1 - FIT["insert_m3_depth"])
                  .pushPoints(polar(CAP_SCREW_R, CAP_SCREW_ANGLES)).circle(FIT["insert_m3_d"] / 2)
                  .extrude(FIT["insert_m3_depth"]))
    cap = cap.cut(cq.Workplane("XY").workplane(offset=z0)
                  .pushPoints(polar(CAP_SCREW_R, [a + 45 for a in CAP_SCREW_ANGLES])).circle(FIT["m3_clear"] / 2)
                  .extrude(z1 - z0))
    cap = cap.cut(cq.Workplane("XY").workplane(offset=z1 - FIT["m3_head_h"])
                  .pushPoints(polar(CAP_SCREW_R, [a + 45 for a in CAP_SCREW_ANGLES])).circle(FIT["m3_head_d"] / 2)
                  .extrude(FIT["m3_head_h"]))
    return cap


# ------------------------------------------------------------ motor can -----
def motor_can(key):
    """Can: front lip (bolts to the cap), motor bore, driver bore.  Prints
    lip-down, open at the back; the rear plate and lid slide in."""
    ms = motor_stack(key)
    lip0, lip1 = S["can_lip"]
    back = ms["back"]
    Ro = ms["outer"]
    can = (cq.Workplane("XY").workplane(offset=back).circle(Ro).extrude(lip1 - back))
    try:
        can = can.faces("<Z").edges().chamfer(1.5)
    except Exception:
        pass
    can = can.cut(cq.Workplane("XY").workplane(offset=lip1 - 0.01 - (lip1 - lip0)).circle(CAP_BORE)
                  .extrude(lip1 - lip0 + 0.02))
    can = can.cut(cq.Workplane("XY").workplane(offset=ms["plate"][0]).circle(ms["bore_motor"])
                  .extrude(lip0 - ms["plate"][0]))
    can = can.cut(cq.Workplane("XY").workplane(offset=back).circle(ms["bore_board"])
                  .extrude(ms["plate"][0] - back))
    # 45 deg transition between bores so the can prints lip-down without support
    if ms["bore_board"] > ms["bore_motor"] + 0.1:
        dz = ms["bore_board"] - ms["bore_motor"]
        cone = (cq.Workplane("XZ").polyline([(0, ms["plate"][0]), (ms["bore_board"], ms["plate"][0]),
                                             (ms["bore_motor"], ms["plate"][0] + dz), (0, ms["plate"][0] + dz)])
                .close().revolve(360, (0, 0, 0), (0, 1, 0)))
        can = can.cut(cone)
    # inserts for the cap screws (in the lip), radial holes for plate + lid screws
    can = can.cut(cq.Workplane("XY").workplane(offset=lip1 - FIT["insert_m3_depth"])
                  .pushPoints(polar(CAP_SCREW_R, [a + 45 for a in CAP_SCREW_ANGLES])).circle(FIT["insert_m3_d"] / 2)
                  .extrude(FIT["insert_m3_depth"]))
    for angs, zc in ((PLATE_SCREWS, sum(ms["plate"]) / 2), (LID_SCREWS, sum(ms["lid"]) / 2)):
        for a in angs:
            d = cq.Vector(math.cos(math.radians(a)), math.sin(math.radians(a)), 0)
            can = can.cut(cq.Workplane("XY").add(cq.Solid.makeCylinder(FIT["m3_clear"] / 2, Ro, cq.Vector(0, 0, zc) + d * 0, d)
                                                  .translate(cq.Vector(0, 0, 0))))
    # motor lead slot along the bore (from motor base to driver)
    can = can.cut(cq.Workplane("XY").workplane(offset=ms["board"][1]).center(0, ms["bore_motor"] - 1)
                  .rect(8, 5).extrude(ms["motor_base"] - ms["board"][1] + 4))
    # thin accent ring
    can = can.cut(cq.Workplane("XZ").polyline([(Ro - 0.6, lip0 - 6), (Ro + 1, lip0 - 6), (Ro + 1, lip0 - 5),
                                               (Ro - 0.6, lip0 - 5)]).close().revolve(360, (0, 0, 0), (0, 1, 0)))
    return can


def rear_plate(key):
    m = MOTORS[key]
    ms = motor_stack(key)
    z0, z1 = ms["plate"]
    r = ms["bore_motor"] - 0.2
    p = cq.Workplane("XY").workplane(offset=z0).circle(r).extrude(z1 - z0)
    n, sd, bc = m["stator_holes"]
    pts = polar(bc / 2, [45 + 360 / n * k for k in range(n)])
    p = p.cut(cq.Workplane("XY").workplane(offset=z0).pushPoints(pts).circle(sd / 2 + 0.2).extrude(z1 - z0))
    p = p.cut(cq.Workplane("XY").workplane(offset=z0).pushPoints(pts).circle(sd + 0.6).extrude(2.5))
    p = p.cut(cq.Workplane("XY").workplane(offset=z0).circle(min(m["bore"] / 2 + 1.0, 8.0)).extrude(z1 - z0))
    p = p.cut(cq.Workplane("XY").workplane(offset=z0).center(0, ms["bore_motor"] - 4).rect(8, 6).extrude(z1 - z0))
    for a in PLATE_SCREWS:
        d = cq.Vector(math.cos(math.radians(a)), math.sin(math.radians(a)), 0)
        p = p.cut(cq.Workplane("XY").add(cq.Solid.makeCylinder(FIT["insert_m3_d"] / 2, FIT["insert_m3_depth"],
                                                               cq.Vector(0, 0, (z0 + z1) / 2) + d * (r - FIT["insert_m3_depth"]), d)))
    # driver standoffs (M2 inserts)
    hs = DRIVER["holes"][0] / 2
    sp = [(hs, hs), (-hs, hs), (-hs, -hs), (hs, -hs)]
    bz0 = ms["board"][1]
    p = p.union(cq.Workplane("XY").workplane(offset=bz0).pushPoints(sp).circle(2.6).extrude(z0 - bz0))
    p = p.cut(cq.Workplane("XY").workplane(offset=bz0).pushPoints(sp).circle(FIT["insert_m2_d"] / 2)
              .extrude(FIT["insert_m2_depth"]))
    return p


def rear_lid(key):
    ms = motor_stack(key)
    z0, z1 = ms["lid"]
    back = ms["back"]
    Ro = ms["outer"]
    lid = cq.Workplane("XY").workplane(offset=z0).circle(ms["bore_board"] - 0.2).circle(ms["bore_board"] - 3.0).extrude(z1 - z0)
    lid = lid.union(cq.Workplane("XY").workplane(offset=back).circle(ms["bore_board"] - 0.2).extrude(z0 - back))
    for a in LID_SCREWS:
        d = cq.Vector(math.cos(math.radians(a)), math.sin(math.radians(a)), 0)
        lid = lid.cut(cq.Workplane("XY").add(cq.Solid.makeCylinder(FIT["insert_m3_d"] / 2, FIT["insert_m3_depth"],
                                                                   cq.Vector(0, 0, (z0 + z1) / 2) + d * (ms["bore_board"] - 0.2 - FIT["insert_m3_depth"]), d)))
    # cable exit + vent slots + accent ring
    lid = lid.cut(cq.Workplane("XY").workplane(offset=back).center(0, -ms["bore_board"] + 5).rect(12, 6).extrude(z1 - back))
    for x in (-8, 0, 8):
        lid = lid.cut(cq.Workplane("XY").workplane(offset=back).center(x, 6).slot2D(14, 2.2, 90).extrude(z0 - back))
    lid = lid.cut(cq.Workplane("XY").workplane(offset=back).circle(17).circle(16).extrude(0.6))
    return lid


# ----------------------------------------------------------- moving parts ---
def hub():
    """Pulley hub: flange + dogs (rear) / bearing journal / capstan drum (front)."""
    f0, f1 = S["hub_flange"]
    r_in = hw.BEARINGS["6705"][0] / 2
    dr0, dr1 = S["drum"]
    fr0, fr1 = S["pul_front"]
    r_drum = CAP["pulley_r"] + 0.1
    neck = 6.4
    pts = [(0, f0), (15.0, f0), (15.0, f1), (r_in, f1), (r_in, S["tube_end"]), (neck, S["tube_end"]),
           (neck, dr0), (r_drum, dr0), (r_drum, dr1), (r_drum + 2.2, dr1 + 2.2), (r_drum + 2.2, fr1), (0, fr1)]
    h = revolve(pts)
    h = h.union(dogs(0.0, S["coupling"][0] + 0.5, f0))
    h = h.cut(cq.Workplane("XY").workplane(offset=S["coupling"][0]).circle(JAW_R[0] - 0.01).extrude(f1 - S["coupling"][0]))
    # retainer inserts in the journal end face
    h = h.cut(cq.Workplane("XY").workplane(offset=S["tube_end"] - FIT["insert_m3_depth"])
              .pushPoints(polar(9.5, [30, 150, 270])).circle(FIT["insert_m3_d"] / 2).extrude(FIT["insert_m3_depth"]))
    # helical cable groove
    pitch, turns = CAP["groove_pitch"], CAP["turns"]
    zc = S["cable_plane"]
    z_start = zc - pitch * turns / 2
    hel = cq.Wire.makeHelix(pitch, pitch * turns, CAP["pulley_r"], center=cq.Vector(0, 0, z_start))
    tang = cq.Vector(0, 2 * math.pi * CAP["pulley_r"], pitch).normalized()
    prof = cq.Workplane(cq.Plane(origin=(CAP["pulley_r"], 0, z_start), xDir=(1, 0, 0), normal=tang)).circle(CAP["cable_d"] / 2 + 0.05)
    h = h.cut(prof.sweep(cq.Workplane().add(hel), isFrenet=True))
    h = h.cut(cq.Workplane("YZ").workplane(offset=-10).center(0, zc).circle(0.8).extrude(20))
    h = h.cut(cq.Workplane("XY").workplane(offset=zc - 2).circle(1.25).extrude(fr1 - zc + 2))
    return h


def rotor_adapter(key):
    m = MOTORS[key]
    a0, a1 = S["adapter"]
    ad = cq.Workplane("XY").workplane(offset=a0).circle(18.0).extrude(a1 - a0)
    ad = ad.union(dogs(60.0, a1, S["coupling"][1] - 0.5))
    n, sd, bc = m["rotor_holes"]
    pts = polar(bc / 2, [45 + 360 / n * k for k in range(n)])
    ad = ad.cut(cq.Workplane("XY").workplane(offset=a0).pushPoints(pts).circle(sd / 2 + 0.2).extrude(a1 - a0))
    ad = ad.cut(cq.Workplane("XY").workplane(offset=a1 - 1.8).pushPoints(pts).circle(sd + 0.6).extrude(1.8))
    return ad


def spider():
    """TPU 95A spider: six legs between the dogs, printed 2 % oversize for preload."""
    c0, c1 = S["coupling"]
    ring = cq.Workplane("XY").workplane(offset=c0 + 0.5).circle(JAW_R[0] + 0.2).circle(JAW_R[0] - 1.5).extrude(c1 - c0 - 1.0)
    for k in range(6):
        c = 60.0 * k + 30.0
        ring = ring.union(sector_ring(JAW_R[0] - 1.5, JAW_R[1] - 0.3, c - LEG_DEG / 2, c + LEG_DEG / 2, c0 + 0.5, c1 - 0.5))
    return ring


def retainer_ring():
    z0, z1 = S["retainer"]
    r = cq.Workplane("XY").workplane(offset=z0).circle(14.2).circle(7.0).extrude(z1 - z0)
    return r.cut(cq.Workplane("XY").workplane(offset=z0).pushPoints(polar(9.5, [30, 150, 270]))
                 .circle(FIT["m3_clear"] / 2).extrude(z1 - z0))


def bearing_spacer():
    z0, z1 = S["brg1"][1], S["lip"][1]
    return cq.Workplane("XY").workplane(offset=z0 + 0.05).circle(14.0).circle(12.55).extrude(z1 - z0 - 0.1)


def magnet_plug(key):
    m = MOTORS[key]
    ms = motor_stack(key)
    rb = m["bore"] / 2 - 0.05
    z_top = ms["motor_base"] + 6.0
    z_end = ms["board"][1] + DRIVER["chip_gap"] + 0.9
    p = cq.Workplane("XY").workplane(offset=z_end).circle(min(rb, 7.5)).extrude(ms["plate"][1] - z_end)
    p = p.union(cq.Workplane("XY").workplane(offset=ms["plate"][1]).circle(rb).extrude(z_top - ms["plate"][1]))
    mr, mh = DRIVER["magnet"]
    p = p.cut(cq.Workplane("XY").workplane(offset=z_end).circle(mr / 2 + 0.05).extrude(mh))
    return p


# ---------------------------------------------------------- assembly -------
_cache = {}


def actuator_parts(key="GL40"):
    if key in _cache:
        return _cache[key]
    m = MOTORS[key]
    ms = motor_stack(key)
    P = []
    P.append(Part("act_front_cap", front_cap(), "printed", COLORS["printed2"], print_down=(0, 0, -1)))
    P.append(Part(f"act_motor_can_{key}", motor_can(key), "printed", COLORS["printed2"], print_down=(0, 0, 1)))
    P.append(Part(f"act_rear_plate_{key}", rear_plate(key), "printed", COLORS["printed2"], print_down=(0, 0, 1)))
    P.append(Part(f"act_rear_lid_{key}", rear_lid(key), "printed", COLORS["printed2"], print_down=(0, 0, -1)))
    P.append(Part("act_hub", hub(), "printed", COLORS["accent"], print_down=(0, 0, -1)))
    P.append(Part(f"act_rotor_adapter_{key}", rotor_adapter(key), "printed", COLORS["accent"], print_down=(0, 0, -1)))
    P.append(Part("act_spider_tpu", spider(), "printed", (0.13, 0.14, 0.16), print_down=(0, 0, -1)))
    P.append(Part("act_retainer_ring", retainer_ring(), "printed", COLORS["accent"], print_down=(0, 0, -1)))
    P.append(Part("act_bearing_spacer", bearing_spacer(), "printed", COLORS["accent"], print_down=(0, 0, -1)))
    P.append(Part(f"act_magnet_plug_{key}", magnet_plug(key), "printed", COLORS["accent"], print_down=(0, 0, 1)))
    # motor
    cols = {"motor_bell": (0.16, 0.17, 0.2), "motor_base": (0.3, 0.31, 0.34), "motor_shaft": COLORS["steel"],
            "motor_stator": COLORS["copper"], "motor_wires": (0.1, 0.1, 0.1)}
    for name, shp in hw.gimbal_motor(m).items():
        P.append(Part(name, shp, "motor", cols[name], M=T(0, 0, ms["motor_base"]),
                      bom=m["name"] if name == "motor_base" else None))
    # driver (chip side faces the motor)
    for name, shp in hw.moteus_c1(DRIVER).items():
        col = {"drv_pcb": (0.05, 0.05, 0.06), "drv_parts": (0.25, 0.25, 0.28), "drv_chip": (0.1, 0.1, 0.1)}[name]
        P.append(Part(name, shp, "pcb", col, M=T(0, 0, ms["board"][1]) @ R((1, 0, 0), 180),
                      bom=DRIVER["name"] + " motor driver" if name == "drv_pcb" else None))
    P.append(Part("magnet_drv", cq.Workplane("XY").circle(DRIVER["magnet"][0] / 2).extrude(DRIVER["magnet"][1]),
                  "hardware", COLORS["magnet"], M=T(0, 0, ms["board"][1] + DRIVER["chip_gap"] + 0.9)))
    # bearings
    for z in (S["brg1"][0], S["brg2"][0]):
        P.append(Part("brg_6705", hw.bearing("6705"), "hardware", COLORS["bearing"], M=T(0, 0, z),
                      bom="6705 bearing (25x32x4)"))
    # fasteners
    ins = "M3 heat-set insert (M3x5.7, 4.6 OD)"
    for x, y in polar(CAP_SCREW_R, [a + 45 for a in CAP_SCREW_ANGLES]):
        P.append(Part("ins_m3", hw.heat_insert("M3"), "hardware", COLORS["brass"], M=T(x, y, S["can_lip"][1]), bom=ins))
        P.append(Part("shcs_m3x10", hw.shcs("M3", 10), "hardware", COLORS["screw"],
                      M=T(x, y, S["cap_ring"][1] - FIT["m3_head_h"]), bom="M3x10 socket head"))
    for x, y in polar(CAP_SCREW_R, CAP_SCREW_ANGLES):
        P.append(Part("ins_m3", hw.heat_insert("M3"), "hardware", COLORS["brass"], M=T(x, y, S["cap_ring"][1]), bom=ins))
    for x, y in polar(9.5, [30, 150, 270]):
        P.append(Part("ins_m3", hw.heat_insert("M3"), "hardware", COLORS["brass"], M=T(x, y, S["tube_end"]), bom=ins))
        P.append(Part("bhcs_m3x6", hw.bhcs("M3", 6), "hardware", COLORS["screw"], M=T(x, y, S["retainer"][1]),
                      bom="M3x6 button head (retainer)"))
    P.append(Part("grub_m3x6", cq.Workplane("XY").circle(1.5).extrude(6), "hardware", COLORS["screw"],
                  M=T(0, 0, S["pul_front"][1] - 6.0), bom="M3x6 grub screw (pulley cable clamp)"))
    for angs, zc, rr in ((PLATE_SCREWS, sum(ms["plate"]) / 2, ms["bore_motor"] - 0.2),
                         (LID_SCREWS, sum(ms["lid"]) / 2, ms["bore_board"] - 0.2)):
        for a in angs:
            c, s_ = math.cos(math.radians(a)), math.sin(math.radians(a))
            Mr = np.eye(4)
            Mr[:3, :3] = np.array([[-s_, 0, c], [c, 0, s_], [0, 1, 0]])
            Mr[:3, 3] = (rr * c, rr * s_, zc)
            P.append(Part("ins_m3", hw.heat_insert("M3"), "hardware", COLORS["brass"], M=Mr, bom=ins))
            Mo = Mr.copy()
            Mo[:3, 3] = (ms["outer"] * c, ms["outer"] * s_, zc)
            P.append(Part("bhcs_m3x6", hw.bhcs("M3", 6), "hardware", COLORS["screw"], M=Mo,
                          bom="M3x6 button head (can)"))
    n, sd, bc = m["stator_holes"]
    sz = "M2.5" if sd < 2.8 else "M3"
    for x, y in polar(bc / 2, [45 + 360 / n * k for k in range(n)]):
        P.append(Part("motor_screw", hw.shcs("M3", 4) if sz == "M3" else hw.shcs("M2", 4), "hardware", COLORS["screw"],
                      M=T(x, y, ms["plate"][0] + 2.5) @ R((1, 0, 0), 180),
                      bom=f"{sz}x6 socket head (motor to plate; check max depth on the drawing)"))
    n, sd, bc = m["rotor_holes"]
    for x, y in polar(bc / 2, [45 + 360 / n * k for k in range(n)]):
        P.append(Part("motor_screw", hw.bhcs("M3", 4), "hardware", COLORS["screw"],
                      M=T(x, y, S["adapter"][1] - 1.8),
                      bom=f"{sz}x5 button head (rotor adapter; check max depth on the drawing)"))
    hs = DRIVER["holes"][0] / 2
    for x, y in [(hs, hs), (-hs, hs), (-hs, -hs), (hs, -hs)]:
        P.append(Part("ins_m2", hw.heat_insert("M2"), "hardware", COLORS["brass"],
                      M=T(x, y, ms["board"][1]) @ R((1, 0, 0), 180), bom="M2 heat-set insert"))
        P.append(Part("shcs_m2x5", hw.shcs("M2", 5), "hardware", COLORS["screw"],
                      M=T(x, y, ms["board"][0]) @ R((1, 0, 0), 180), bom="M2x5 socket head (driver)"))
    _cache[key] = P
    return P


def pulley_cable_wrap():
    pitch, turns = CAP["groove_pitch"], CAP["turns"]
    z_start = S["cable_plane"] - pitch * turns / 2
    return hw.cable_helix(CAP["pulley_r"], pitch, turns, z_start, CAP["cable_d"])
