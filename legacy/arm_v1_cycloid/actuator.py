"""Cycloidal + capstan actuator module.

Actuator frame: +Z is the motor/gearbox axis, Z=0 is the motor mounting face,
the capstan pulley is at the +Z end.  The front-cap face (Z = mount_face) bolts
flat against a fork cheek; the same 4 screws hold the cap to the housing.
"""
import math
import numpy as np
import cadquery as cq

from .params import (MOTOR, FIT, CYC, CAP, STACK as S, HOUSING_R, CAN_R,
                     PLATE_HALF, CAP_SCREW_R, TUBE_R, ENC)
from . import hardware as hw
from .core import Part, COLORS, T, R

BRG_SEAT = lambda name: hw.BEARINGS[name][1] / 2 + FIT["bearing_seat"]
CAP_SCREW_ANGLES = [45, 135, 225, 315]
CARRIER_R = 17.5
CAP_RING_IR = 18.0


def revolve(pts):
    """Revolve an (r, z) outline about Z."""
    return (cq.Workplane("XZ").polyline(pts).close()
            .revolve(360, (0, 0, 0), (0, 1, 0)))


def polar(r, angles):
    return [(r * math.cos(math.radians(a)), r * math.sin(math.radians(a)))
            for a in angles]


# ------------------------------------------------------------- cycloid ------
def cycloid_profile(n=720):
    R_, E, N = CYC["pin_circle_r"], CYC["ecc"], CYC["n_pins"]
    Rr = CYC["pin_d"] / 2 + CYC["profile_clear"]
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    psi = np.arctan2(np.sin((1 - N) * t), (R_ / (E * N)) - np.cos((1 - N) * t))
    x = R_ * np.cos(t) - Rr * np.cos(t + psi) - E * np.cos(N * t)
    y = -R_ * np.sin(t) + Rr * np.sin(t + psi) + E * np.sin(N * t)
    # profile is centred on the disc centre; mesh verified with disc at (+E, 0)
    return [(float(a), float(b)) for a, b in zip(x, y)]


def cycloid_disc():
    pts = cycloid_profile()
    disc = cq.Workplane("XY").spline(pts, periodic=True).close().extrude(CYC["disc_t"])
    disc = disc.cut(cq.Workplane("XY").circle(BRG_SEAT("6800")).extrude(CYC["disc_t"]))
    hole_d = CYC["out_pin_d"] + 2 * CYC["ecc"] + 2 * CYC["out_hole_clear"]
    holes = (cq.Workplane("XY").pushPoints(polar(CYC["out_pin_r"],
             [360 / CYC["n_out"] * k for k in range(CYC["n_out"])]))
             .circle(hole_d / 2).extrude(CYC["disc_t"]))
    disc = disc.cut(holes)
    # small index notch so both discs can be assembled 180 deg apart
    notch = (cq.Workplane("XY").center(*polar(18.0, [30])[0]).circle(0.6).extrude(CYC["disc_t"]))
    return disc.cut(notch)


# ------------------------------------------------------------- housing ------
END_CAP_H = 9.0
END_CAP_SCREWS = [(0.0, 24.0), (0.0, -24.0)]


def motor_can():
    """Rear housing: motor mount plate + motor bay.  Prints plate-down with an
    open top.  A 1 mm spigot on top centres the ring housing."""
    z_wall0 = S["wall"][0]
    z_back = S["plate"][0]
    R_ = HOUSING_R
    body = (cq.Workplane("XY").workplane(offset=z_back).circle(R_).extrude(z_wall0 - z_back)
            .faces("<Z").edges().chamfer(1.2))
    body = body.union(cq.Workplane("XY").workplane(offset=z_wall0).circle(R_)
                      .circle(SPIGOT_R).extrude(1.0))
    can_r = MOTOR["bell_d"] / 2 + 1.0
    body = body.cut(cq.Workplane("XY").circle(can_r).extrude(z_wall0 + 1))
    plate_t = -z_back
    wp = lambda: cq.Workplane("XY").workplane(offset=z_back)
    body = body.cut(wp().circle(5.0).extrude(plate_t))
    h = MOTOR["mount_square"] / 2
    mpts = [(h, h), (-h, h), (-h, -h), (h, -h)]
    body = body.cut(wp().pushPoints(mpts).circle(FIT["m3_clear"] / 2).extrude(plate_t))
    body = body.cut(wp().pushPoints(mpts).circle(FIT["m3_head_d"] / 2).extrude(FIT["m3_head_h"]))
    body = body.cut(wp().center(0, 14.5).rect(9, 4).extrude(plate_t))
    hs = ENC["board_holes"] / 2
    body = body.cut(wp().pushPoints([(hs, 0), (-hs, 0)]).circle(FIT["insert_m2_d"] / 2)
                    .extrude(FIT["insert_m2_depth"]))
    body = body.cut(wp().pushPoints(END_CAP_SCREWS).circle(FIT["insert_m3_d"] / 2)
                    .extrude(FIT["insert_m3_depth"]))
    # inserts for the 4 through-bolts (cheek -> cap -> ring housing -> here)
    body = body.cut(cq.Workplane("XY").workplane(offset=z_wall0 - FIT["insert_m3_depth"])
                    .pushPoints(polar(CAP_SCREW_R, CAP_SCREW_ANGLES)).circle(FIT["insert_m3_d"] / 2)
                    .extrude(FIT["insert_m3_depth"]))
    return body


SPIGOT_R = 29.4   # outer locating rim (can) / rebate (ring housing)


def ring_housing():
    """Ring-gear housing: rear wall + ring-pin seats.  Prints wall-down."""
    z_wall0, z_wall1 = S["wall"]
    z_front = S["cap_ring"][0]
    R_ = HOUSING_R
    body = (cq.Workplane("XY").workplane(offset=z_wall0).circle(R_).extrude(z_front - z_wall0)
            .faces(">Z").edges().chamfer(0.6))
    body = body.cut(cq.Workplane("XY").workplane(offset=z_wall0).circle(R_ + 1)
                    .circle(SPIGOT_R - 0.1).extrude(1.0))
    body = body.cut(cq.Workplane("XY").workplane(offset=z_wall0).circle(7.0).extrude(z_wall1 - z_wall0))
    gb_r = CYC["pin_circle_r"] + 2.0
    body = body.cut(cq.Workplane("XY").workplane(offset=z_wall1).circle(gb_r).extrude(z_front - z_wall1))
    pin_hole = CYC["pin_d"] / 2 - FIT["press"]
    pins = polar(CYC["pin_circle_r"], [360 / CYC["n_pins"] * k for k in range(CYC["n_pins"])])
    body = body.cut(cq.Workplane("XY").workplane(offset=z_wall0).pushPoints(pins)
                    .circle(pin_hole).extrude(z_wall1 - z_wall0))
    body = body.cut(cq.Workplane("XY").workplane(offset=z_wall0)
                    .pushPoints(polar(CAP_SCREW_R, CAP_SCREW_ANGLES)).circle(FIT["m3_clear"] / 2)
                    .extrude(z_front - z_wall0))
    return body


def end_cap():
    """Rear cover over the motor encoder + wiring.  Occupies z in
    [plate_back - END_CAP_H, plate_back]; prints top-down with no supports."""
    z1 = S["plate"][0]
    z0 = z1 - END_CAP_H
    R_ = HOUSING_R
    wall, top = 1.6, 1.6
    cap = (cq.Workplane("XY").workplane(offset=z0).circle(R_).extrude(END_CAP_H)
           .faces("<Z").edges().chamfer(2.0))
    cap = cap.cut(cq.Workplane("XY").workplane(offset=z0 + top).circle(R_ - wall)
                  .extrude(END_CAP_H - top))
    cap = cap.union(cq.Workplane("XY").workplane(offset=z0 + top).pushPoints(END_CAP_SCREWS)
                    .circle(4.0).extrude(END_CAP_H - top))
    cap = cap.cut(cq.Workplane("XY").workplane(offset=z0).pushPoints(END_CAP_SCREWS)
                  .circle(FIT["m3_clear"] / 2).extrude(END_CAP_H))
    cap = cap.cut(cq.Workplane("XY").workplane(offset=z0).pushPoints(END_CAP_SCREWS)
                  .circle(FIT["m3_head_d"] / 2).extrude(FIT["m3_head_h"]))
    # accent ring groove + cable exit notch
    cap = cap.cut(cq.Workplane("XY").workplane(offset=z0).circle(17.0).circle(16.0).extrude(0.6))
    cap = cap.cut(cq.Workplane("XY").workplane(offset=z1 - 4.0).center(R_ - 1, 0)
                  .rect(6, 10).extrude(4.0))
    return cap


def front_cap():
    z0, z1 = S["cap_ring"]
    s1 = BRG_SEAT("6705")
    pts = [(CAP_RING_IR, z0), (HOUSING_R, z0), (HOUSING_R, z1), (TUBE_R, z1),
           (TUBE_R, S["tube_end"]), (s1, S["tube_end"]), (s1, S["lip"][1]),
           (14.0, S["lip"][1]), (14.0, S["lip"][0] + (s1 - 14.0)), (s1, S["lip"][0]),
           (s1, z1), (CAP_RING_IR, z1)]
    cap = revolve(pts)
    try:
        cap = cap.faces(">Z").edges(cq.selectors.RadiusNthSelector(-1)).chamfer(0.6)
    except Exception:
        pass
    pin_hole = CYC["pin_d"] / 2 - FIT["press"]
    pins = polar(CYC["pin_circle_r"], [360 / CYC["n_pins"] * k for k in range(CYC["n_pins"])])
    cap = cap.cut(cq.Workplane("XY").workplane(offset=z0).pushPoints(pins)
                  .circle(pin_hole).extrude(S["ring_pin"][1] - z0 + 0.3))
    cap = cap.cut(cq.Workplane("XY").workplane(offset=z0)
                  .pushPoints(polar(CAP_SCREW_R, CAP_SCREW_ANGLES))
                  .circle(FIT["m3_clear"] / 2).extrude(z1 - z0))
    return cap


def eccentric_hub():
    E = CYC["ecc"]
    d1, d2 = S["disc1"], S["disc2"]
    r_ecc = hw.BEARINGS["6800"][0] / 2
    r_stub = hw.BEARINGS["6801"][0] / 2
    hub = cq.Workplane("XY").workplane(offset=S["hub"][0]).circle(6.0).extrude(d1[0] - S["hub"][0])
    hub = hub.union(cq.Workplane("XY").workplane(offset=d1[0]).center(E, 0)
                    .circle(r_ecc).extrude(d1[1] - d1[0]))
    hub = hub.union(cq.Workplane("XY").workplane(offset=d1[1]).circle(6.0)
                    .extrude(d2[0] - d1[1]))
    hub = hub.union(cq.Workplane("XY").workplane(offset=d2[0]).center(-E, 0)
                    .circle(r_ecc).extrude(d2[1] - d2[0]))
    hub = hub.union(cq.Workplane("XY").workplane(offset=d2[1]).circle(r_stub)
                    .extrude(S["hub"][1] - d2[1]))
    hub = hub.cut(cq.Workplane("XY").workplane(offset=S["hub"][0])
                  .circle(FIT["tap_m5"] / 2).extrude(S["hub"][1] - S["hub"][0]))
    return hub


def output_carrier():
    """Output plate + hub + capstan drum in one piece."""
    z0, z1 = S["carrier_plate"]
    seat = BRG_SEAT("6801")
    sh0, sh1 = S["shoulder"]
    dr0, dr1 = S["drum"]
    f0, f1 = S["pul_front"]
    r_drum = CAP["pulley_r"] + 0.1
    r_inner = hw.BEARINGS["6705"][0] / 2
    pts = [(seat, z0), (CARRIER_R, z0), (CARRIER_R, z1), (14.0, z1), (14.0, sh1),
           (r_inner, sh1), (r_inner, S["tube_end"]), (r_drum, S["tube_end"]),
           (r_drum, dr1), (9.0, dr1 + (9.0 - r_drum)), (9.0, f1), (0, f1),
           (0, z1 + 0.5), (9.0, z1 + 0.5), (9.0, z1), (seat, z1)]
    car = revolve(pts)
    # output pins (press fit, through the plate)
    pin_hole = CYC["out_pin_d"] / 2 - FIT["press"]
    car = car.cut(cq.Workplane("XY").workplane(offset=z0)
                  .pushPoints(polar(CYC["out_pin_r"], [360 / CYC["n_out"] * k for k in range(CYC["n_out"])]))
                  .circle(pin_hole).extrude(z1 - z0))
    # inserts for retainer ring screws
    car = car.cut(cq.Workplane("XY").workplane(offset=S["tube_end"] - FIT["insert_m3_depth"])
                  .pushPoints(polar(9.5, [30, 150, 270])).circle(FIT["insert_m3_d"] / 2)
                  .extrude(FIT["insert_m3_depth"]))
    # helical cable groove
    pitch, turns = CAP["groove_pitch"], CAP["turns"]
    zc = S["cable_plane"]
    z_start = zc - pitch * turns / 2
    hel = cq.Wire.makeHelix(pitch, pitch * turns, CAP["pulley_r"],
                            center=cq.Vector(0, 0, z_start))
    tang = cq.Vector(0, 2 * math.pi * CAP["pulley_r"], pitch).normalized()
    prof = cq.Workplane(cq.Plane(origin=(CAP["pulley_r"], 0, z_start),
                                 xDir=(1, 0, 0), normal=tang)).circle(CAP["cable_d"] / 2 + 0.05)
    groove = prof.sweep(cq.Workplane().add(hel), isFrenet=True)
    car = car.cut(groove)
    # cable cross hole + axial M3 clamp screw (tap 2.5)
    car = car.cut(cq.Workplane("YZ").workplane(offset=-10).center(0, zc)
                  .circle(0.8).extrude(20))
    car = car.cut(cq.Workplane("XY").workplane(offset=zc - 2).circle(1.25)
                  .extrude(f1 - zc + 2))
    return car


def retainer_ring():
    z0, z1 = S["retainer"]
    r = (cq.Workplane("XY").workplane(offset=z0).circle(14.2).circle(7.0).extrude(z1 - z0))
    return r.cut(cq.Workplane("XY").workplane(offset=z0)
                 .pushPoints(polar(9.5, [30, 150, 270])).circle(FIT["m3_clear"] / 2)
                 .extrude(z1 - z0))


# ---------------------------------------------------------- assembly -------
_cache = {}


def actuator_parts():
    """List of Parts in the actuator frame.  Cached (same for all joints)."""
    if "parts" in _cache:
        return _cache["parts"]
    P = []
    down = (0, 0, -1)
    P.append(Part("act_motor_can", motor_can(), "printed", COLORS["printed2"], print_down=(0, 0, -1)))
    P.append(Part("act_ring_housing", ring_housing(), "printed", COLORS["printed2"], print_down=(0, 0, -1)))
    P.append(Part("act_front_cap", front_cap(), "printed", COLORS["printed2"], print_down=(0, 0, -1)))
    P.append(Part("act_eccentric_hub", eccentric_hub(), "printed", COLORS["accent"], print_down=(0, 0, -1)))
    disc = cycloid_disc()
    P.append(Part("act_cycloid_disc", disc, "printed", COLORS["accent"],
                  M=T(CYC["ecc"], 0, S["disc1"][0]), print_down=down))
    P.append(Part("act_cycloid_disc", disc, "printed", COLORS["accent"],
                  M=T(-CYC["ecc"], 0, S["disc2"][0]) @ R((0, 0, 1), 180), print_down=down))
    P.append(Part("act_output_carrier", output_carrier(), "printed", COLORS["accent"], print_down=down))
    P.append(Part("act_retainer_ring", retainer_ring(), "printed", COLORS["accent"], print_down=down))
    P.append(Part("act_end_cap", end_cap(), "printed", COLORS["printed2"], print_down=(0, 0, -1)))
    for x, y in END_CAP_SCREWS:
        P.append(Part("ins_m3", hw.heat_insert("M3"), "hardware", COLORS["brass"],
                      M=T(x, y, S["plate"][0]) @ R((1, 0, 0), 180), bom="M3 heat-set insert (M3x5.7, 4.6 OD)"))
        P.append(Part("shcs_m3x10", hw.shcs("M3", 10), "hardware", COLORS["screw"],
                      M=T(x, y, S["plate"][0] - END_CAP_H + FIT["m3_head_h"]) @ R((1, 0, 0), 180),
                      bom="M3x10 socket head"))

    # motor
    for name, shp in hw.motor().items():
        col = {"motor_bell": COLORS["motor_bell"], "motor_stator": COLORS["copper"],
               "motor_shaft": COLORS["steel"], "motor_wires": (0.1, 0.1, 0.1)}.get(name, COLORS["motor"])
        P.append(Part(name, shp, "motor", col,
                      bom=("Drone motor " + MOTOR["name"]) if name == "motor_base" else None))
    # motor screws (from the rear, heads in counterbores)
    h = MOTOR["mount_square"] / 2
    scr_len = (-S["plate"][0] - FIT["m3_head_h"]) + MOTOR["mount_thread_depth"]
    scr_len = math.floor(scr_len)
    for x, y in [(h, h), (-h, h), (-h, -h), (h, -h)]:
        P.append(Part("m3_motor", hw.shcs("M3", scr_len), "hardware", COLORS["screw"],
                      M=T(x, y, S["plate"][0] + FIT["m3_head_h"]) @ R((1, 0, 0), 180),
                      bom=f"M3x{scr_len} socket head (motor; do NOT go longer)"))
    # bearings
    def brg(name, z, x=0.0):
        P.append(Part("brg_" + name, hw.bearing(name), "hardware", COLORS["bearing"],
                      M=T(x, 0, z), bom=f"{name} bearing {hw.BEARINGS[name]}"))
    brg("6800", S["disc1"][0], CYC["ecc"])
    brg("6800", S["disc2"][0], -CYC["ecc"])
    brg("6801", S["carrier_plate"][0])
    brg("6705", S["brg1"][0])
    brg("6705", S["brg2"][0])
    # ring pins
    L = S["ring_pin"][1] - S["ring_pin"][0]
    for x, y in polar(CYC["pin_circle_r"], [360 / CYC["n_pins"] * k for k in range(CYC["n_pins"])]):
        P.append(Part("ring_pin", hw.dowel(CYC["pin_d"], L), "hardware", COLORS["steel"],
                      M=T(x, y, S["ring_pin"][0]), bom=f"Dowel pin {CYC['pin_d']:g}x{L:g} (ring pin)"))
    L = S["out_pin"][1] - S["out_pin"][0]
    for x, y in polar(CYC["out_pin_r"], [360 / CYC["n_out"] * k for k in range(CYC["n_out"])]):
        P.append(Part("out_pin", hw.dowel(CYC["out_pin_d"], L), "hardware", COLORS["steel"],
                      M=T(x, y, S["out_pin"][0]), bom=f"Dowel pin {CYC['out_pin_d']:g}x{L:g} (output pin)"))
    # inserts in housing
    for x, y in polar(CAP_SCREW_R, CAP_SCREW_ANGLES):
        P.append(Part("ins_m3", hw.heat_insert("M3"), "hardware", COLORS["brass"],
                      M=T(x, y, S["wall"][0]), bom="M3 heat-set insert (M3x5.7, 4.6 OD)"))
    # retainer screws + inserts
    for x, y in polar(9.5, [30, 150, 270]):
        P.append(Part("ins_m3", hw.heat_insert("M3"), "hardware", COLORS["brass"],
                      M=T(x, y, S["tube_end"]), bom="M3 heat-set insert (M3x5.7, 4.6 OD)"))
        P.append(Part("bhcs_m3x6", hw.bhcs("M3", 6), "hardware", COLORS["screw"],
                      M=T(x, y, S["retainer"][1]), bom="M3x6 button head (retainer)"))
    # cable clamp grub screw
    P.append(Part("grub_m3x6", cq.Workplane("XY").circle(1.5).extrude(6), "hardware",
                  COLORS["screw"], M=T(0, 0, S["pul_front"][1] - 6.0),
                  bom="M3x6 grub screw (pulley cable clamp)"))
    # motor encoder: magnet on rear shaft stub + board on the rear plate
    mag_z = -MOTOR["rear_shaft_stub"]
    P.append(Part("magnet", hw.magnet(), "hardware", COLORS["magnet"],
                  M=T(0, 0, mag_z) @ R((1, 0, 0), 180),
                  bom="Diametric magnet 6x2.5 mm"))
    P.append(Part("enc_board", hw.encoder_board(), "pcb", COLORS["pcb"],
                  M=T(0, 0, S["plate"][0]) @ R((1, 0, 0), 180),
                  bom="Encoder breakout (MT6701 / AS5047P), 24 mm M2 holes"))
    for x in (-ENC["board_holes"] / 2, ENC["board_holes"] / 2):
        P.append(Part("ins_m2", hw.heat_insert("M2"), "hardware", COLORS["brass"],
                      M=T(x, 0, S["plate"][0]) @ R((1, 0, 0), 180),
                      bom="M2 heat-set insert"))
        P.append(Part("shcs_m2x5", hw.shcs("M2", 5), "hardware", COLORS["screw"],
                      M=T(x, 0, S["plate"][0] - ENC["board"][2]) @ R((1, 0, 0), 180),
                      bom="M2x5 socket head (encoder)"))
    _cache["parts"] = P
    return P


def pulley_cable_wrap():
    """Cable turns on the pulley (actuator frame)."""
    pitch, turns = CAP["groove_pitch"], CAP["turns"]
    z_start = S["cable_plane"] - pitch * turns / 2
    return hw.cable_helix(CAP["pulley_r"], pitch, turns, z_start, CAP["cable_d"])
