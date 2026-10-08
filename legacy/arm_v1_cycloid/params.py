"""
All dimensions in millimetres.  EDIT THE "MOTOR" SECTION FIRST.

Everything else (housing, gearbox, pulley, sectors, links) is derived from
these numbers, so after changing the motor, rerun `python build.py` and the
whole arm regenerates around it.
"""
import math

# ============================================================================
# MOTOR  --  measure your drone motor with calipers and edit these
# ============================================================================
MOTOR = dict(
    name="2207 FPV motor (placeholder)",
    kv=1900,                # rpm/V  -> torque constant estimate
    bell_d=27.9,            # outer diameter of the spinning bell
    body_h=20.0,            # mounting face of base -> top of bell (excl. shaft)
    base_d=26.0,            # diameter of the fixed base/mount
    mount_square=16.0,      # mounting holes on a square of this side (16x16 typical)
    mount_screw="M3",
    mount_thread_depth=3.5, # max safe screw depth into the base (check yours!)
    shaft_d=5.0,            # prop shaft (M5 on most 22xx/23xx)
    shaft_len=13.0,         # how far the shaft sticks out above the bell
    rear_shaft_stub=1.0,    # shaft end + circlip protruding below the base
    stator_d=22.0,          # cosmetic only
    stator_h=7.0,           # cosmetic only
    mass_g=33,
)

# ============================================================================
# PRINT FIT  --  tune these to your printer
# ============================================================================
FIT = dict(
    clearance=0.25,      # general sliding clearance (radial)
    press=0.05,          # press-fit undersize for dowel holes (radial)
    bearing_seat=0.05,   # radial clearance for bearing outer races
    insert_m3_d=4.0,     # hole for M3 heat-set insert (check insert datasheet)
    insert_m3_depth=5.5,
    insert_m2_d=3.2,
    insert_m2_depth=4.0,
    m3_clear=3.4,
    m3_head_d=6.0,       # counterbore for socket head
    m3_head_h=3.2,
    m4_clear=4.4,
    tap_m5=4.2,          # hub bore -> tap M5 (or thread straight onto the prop shaft)
)

# ============================================================================
# CYCLOIDAL STAGE
# ============================================================================
CYC = dict(
    n_pins=16,           # ring pins -> ratio = n_pins - 1
    pin_circle_r=22.0,   # ring-pin circle radius R
    pin_d=3.0,           # ring pin = 3 mm steel dowel
    ecc=0.9,             # eccentricity E
    profile_clear=0.05,  # extra pin radius used when generating the profile
    disc_t=5.0,          # each disc (matches 6800 bearing width)
    n_out=6,             # output pins
    out_pin_r=14.0,      # output pin circle radius
    out_pin_d=3.0,       # output pin = 3 mm steel dowel
    out_hole_clear=0.05, # radial clearance in the disc's output holes (ream to size: 3 + 2E + 2c)
)
CYC["ratio"] = CYC["n_pins"] - 1

# ============================================================================
# CAPSTAN STAGE
# ============================================================================
CAP = dict(
    cable_d=1.0,         # 1 mm Dyneema (or 0.9 mm 7x7 steel)
    pulley_r=6.5,        # cable centre radius on the pulley
    groove_pitch=1.3,
    turns=4.5,
    gap=1.0,             # extra centre-distance so drum and sector never touch
    sector_t=7.0,        # sector rim thickness (axial)
    anchor_margin=15.0,  # degrees of rim beyond the range of motion at each end
)

# ============================================================================
# ARM LAYOUT
# ============================================================================
ARM = dict(
    L1=150.0,            # shoulder -> elbow
    L2=130.0,            # elbow -> wrist pitch
    base_size=190.0,
    base_h=72.0,
    top_plate_t=8.0,
    cheek_t=8.0,         # fork cheek plate thickness
    interior_half=14.0,  # half-width between fork cheeks
    body_half=13.5,      # half-width of link bodies
    hub_r=16.0,          # child hub radius at each joint
    pad_r=26.0,          # sector mounting pad radius
    joint_shaft_d=8.0,   # 8 mm steel shafts in 608 bearings
    beam_h=13.0,         # half-height of link beams
)

# cable-centre radius on each joint's sector (sets the capstan ratio)
#            Rc     ROM (deg)       pulley direction (deg, in parent frame)  side
JOINTS = {
    "J1": dict(Rc=47.0, rom=(-150, 150), dir=0.0,   side=+1, name="base yaw"),
    "J2": dict(Rc=52.0, rom=(-10, 120),  dir=180.0, side=+1, name="shoulder"),
    "J3": dict(Rc=40.0, rom=(-110, 110), dir=180.0, side=-1, name="elbow"),
    "J4": dict(Rc=32.5, rom=(-90, 70),   dir=180.0, side=+1, name="wrist pitch"),
    "J5": dict(Rc=30.0, rom=(-150, 150), dir=90.0,  side=+1, name="wrist roll"),
}
for j in JOINTS.values():
    j["d"] = j["Rc"] + CAP["pulley_r"] + CAP["gap"]           # pulley offset
    j["cap_ratio"] = j["Rc"] / CAP["pulley_r"]
    j["ratio"] = j["cap_ratio"] * CYC["ratio"]
    lo, hi = j["rom"]
    m = j.get("margin", CAP["anchor_margin"])
    j["arc"] = (j["dir"] - hi - m, j["dir"] - lo + m)          # sector arc in child frame

# Wrist roll + parallel gripper (hand frame: J4 at origin, x forward, z up)
GRIP = dict(
    yoke_x=82.0,         # rear face of the wrist yoke (roll actuator mounts behind it)
    link_L=24.0,         # parallelogram link length (pivot to pivot)
    link_h=14.0,         # spacing of the two pivots on each finger
    pivot_z=6.0,         # driven-link pivots at z = +/-pivot_z (= sync gear pitch radius)
    pivot_x=34.0,        # pivot line, measured from the palm's rear face
    phi=(0.0, 50.0),     # link swing: 0 = closed (1 mm gap), 50 = fully open
    face=5.5,            # finger pad face sits this far below the lower finger pin
    gear_module=1.0,
    servo=dict(name="MG90S metal-gear micro servo", body=(22.8, 12.2, 22.7), tab_len=32.2,
               tab_h=(15.9, 18.4), tab_holes=28.0, shaft_off=5.9, boss_h=4.3, spline_h=3.0,
               stall_nm=0.21),
)

# ============================================================================
# ACTUATOR AXIAL STACK (derived) -- z along actuator axis, 0 = motor mount face
# ============================================================================
def actuator_stack():
    M, C = MOTOR, CYC
    s = {}
    s["plate"] = (-6.0, 0.0)
    s["bell_top"] = M["body_h"]
    s["wall"] = (M["body_h"] + 1.0, M["body_h"] + 5.0)
    z = s["wall"][1] + 0.5
    s["disc1"] = (z, z + C["disc_t"]); z += C["disc_t"] + 0.5
    s["disc2"] = (z, z + C["disc_t"]); z += C["disc_t"] + 0.5
    s["carrier_plate"] = (z, z + 5.0)
    s["cap_ring"] = (z, z + 5.0)
    s["mount_face"] = z + 5.0
    z += 5.0
    s["shoulder"] = (z, z + 0.5); z += 0.5
    s["brg1"] = (z, z + 4.0); z += 4.0
    s["lip"] = (z, z + 3.0); z += 3.0
    s["brg2"] = (z, z + 4.0); z += 4.0
    s["tube_end"] = z
    s["carrier_hub"] = (s["carrier_plate"][1], z)
    s["retainer"] = (z, z + 1.0)              # printed ring clamps bearing-2 inner race
    s["retainer_screw_top"] = z + 1.0 + 1.65  # M3 button heads
    z += 1.0 + 1.65 + 0.5
    drum_len = CAP["sector_t"] + 1.0
    s["drum"] = (z, z + drum_len); z += drum_len
    s["pul_front"] = (z, z + 3.0); z += 3.0   # 45 deg cone + 0.6 lip (prints without support)
    s["cable_plane"] = (s["drum"][0] + s["drum"][1]) / 2
    s["ring_pin"] = (s["wall"][0], s["cap_ring"][0] + 4.5)
    s["out_pin"] = (s["disc1"][0], s["carrier_plate"][1])
    s["hub"] = (M["body_h"] + 0.5, s["carrier_plate"][0] + 4.7)
    s["end"] = z
    return s

STACK = actuator_stack()

# Joint-plane mapping: |y| = CHEEK_OUT - (z_act - mount_face)
CHEEK_OUT = ARM["interior_half"] + ARM["cheek_t"]           # 22
def act_to_abs_y(z):
    return CHEEK_OUT - (z - STACK["mount_face"])
SECTOR_Y = (act_to_abs_y(STACK["cable_plane"]) - CAP["sector_t"] / 2,
            act_to_abs_y(STACK["cable_plane"]) + CAP["sector_t"] / 2)

HOUSING_R = 31.0      # gearbox ring outer radius
CAN_R = 18.0          # motor-can outer radius
PLATE_HALF = 33.0     # rear plate half-size
CAP_SCREW_R = 27.5    # cap/mounting screw circle
TUBE_R = 19.5         # front bearing tube outer radius

# Encoder mounting (same pattern everywhere)
ENC = dict(
    board=(30.0, 18.0, 1.6),    # generic MT6701/AS5047 breakout (edit to suit)
    board_holes=24.0,           # M2 hole spacing on the board
    mount_holes=36.0,           # M3 spacing of the printed bracket feet
    magnet=(6.0, 2.5),          # diametric magnet d x h
    gap=1.5,                    # magnet face -> chip
    chip_t=1.0,
)

def kt_nm_per_a():
    return 8.27 / MOTOR["kv"]
