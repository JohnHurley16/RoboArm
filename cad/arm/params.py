"""
All dimensions in millimetres.  EDIT THE "MOTORS" SECTION FIRST.

Each joint is a low-KV gimbal motor driving a capstan directly (no gearbox),
through a zero-backlash jaw coupling.  Everything else is derived from these
numbers: rerun `python build.py` after editing.
"""
import math

# ============================================================================
# MOTORS  --  CubeMars gimbal motors (specs from cubemars.com, Oct 2026)
# Mounting-hole patterns are PLACEHOLDERS: check CubeMars' 2D drawing for your
# motor and edit rotor_holes / stator_holes before printing.
# ============================================================================
MOTORS = {
    "GL40": dict(name="CubeMars GL40 KV70", od=46.5, h=21.5, bore=8.0,
                 rotor_holes=(4, 2.5, 16.0),    # count, screw size, bolt circle  (PLACEHOLDER)
                 stator_holes=(4, 2.5, 25.0),   # count, screw size, bolt circle  (PLACEHOLDER)
                 kt=0.150, rated_nm=0.25, peak_nm=0.5, rated_a=1.62, peak_a=3.3, r_ohm=4.5,
                 mass_g=107, price=72.99,
                 url="https://www.cubemars.com/product/gl40-kv70-gimbal-motor.html"),
    "GL60": dict(name="CubeMars GL60 KV25", od=69.0, h=22.3, bore=20.0,
                 rotor_holes=(4, 3.0, 30.0),    # PLACEHOLDER
                 stator_holes=(4, 3.0, 32.0),   # PLACEHOLDER
                 kt=0.450, rated_nm=0.6, peak_nm=1.75, rated_a=1.35, peak_a=4.0, r_ohm=5.5,
                 mass_g=230, price=108.99,
                 url="https://www.cubemars.com/product/gl60-kv25-gimbal-motor.html"),
}

DRIVER = dict(name="mjbots moteus-c1", board=(38.0, 38.0, 9.0), mass_g=15, price=69.0,
              holes=(30.0, 2.0),                # square spacing, screw size  (PLACEHOLDER: check STEP)
              magnet=(6.35, 2.5),               # 1/4" diametric magnet shipped with moteus
              chip_gap=1.5,
              url="https://mjbots.com/products/moteus-c1")

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
# CAPSTAN STAGE
# ============================================================================
CAP = dict(
    cable_d=1.0,         # 1 mm Dyneema (or 0.9 mm 7x7 steel)
    pulley_r=5.0,        # cable centre radius on the pulley (10x cable diameter)
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
    "J1": dict(motor="GL40", Rc=48.5, rom=(-150, 150), dir=0.0,   side=+1, name="base yaw"),
    "J2": dict(motor="GL60", Rc=52.0, rom=(-10, 120),  dir=180.0, side=+1, name="shoulder"),
    "J3": dict(motor="GL60", Rc=40.0, rom=(-110, 110), dir=180.0, side=-1, name="elbow"),
    "J4": dict(motor="GL40", Rc=32.5, rom=(-90, 70),   dir=180.0, side=+1, name="wrist pitch"),
    "J5": dict(motor="GL40", Rc=30.0, rom=(-150, 150), dir=90.0,  side=+1, name="wrist roll"),
}
for j in JOINTS.values():
    j["d"] = j["Rc"] + CAP["pulley_r"] + CAP["gap"]           # pulley offset
    j["cap_ratio"] = j["Rc"] / CAP["pulley_r"]
    j["ratio"] = j["cap_ratio"]
    lo, hi = j["rom"]
    m = j.get("margin", CAP["anchor_margin"])
    j["arc"] = (j["dir"] - hi - m, j["dir"] - lo + m)          # sector arc in child frame

# Wrist roll + parallel gripper (hand frame: J4 at origin, x forward, z up)
GRIP = dict(
    yoke_x=88.0,         # rear face of the wrist yoke (roll actuator mounts behind it)
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
    """Motor-independent front stack.  z = 0 is the mounting face (fork cheek
    outer face); +z points into the joint.  The motor sits behind (z < 0)."""
    s = {}
    s["mount_face"] = 0.0
    s["cap_ring"] = (-8.0, 0.0)              # front-cap flange, outside the cheek
    s["can_lip"] = (-14.5, -8.0)             # motor-can front lip (cap screws thread in here)
    s["hub_flange"] = (-2.0, 0.5)
    s["coupling"] = (-12.5, -2.0)            # jaw coupling (hub dogs / TPU spider / rotor dogs)
    s["adapter"] = (-15.0, -12.5)            # rotor adapter disc, bolted to the rotor top
    s["rotor_top"] = -15.0
    z = 0.0
    s["shoulder"] = (z, z + 0.5); z += 0.5
    s["brg1"] = (z, z + 4.0); z += 4.0
    s["lip"] = (z, z + 3.0); z += 3.0
    s["brg2"] = (z, z + 4.0); z += 4.0
    s["tube_end"] = z
    s["retainer"] = (z, z + 1.0)
    s["retainer_screw_top"] = z + 1.0 + 1.65
    z += 1.0 + 1.65 + 0.5
    drum_len = CAP["sector_t"] + 1.0
    s["drum"] = (z, z + drum_len); z += drum_len
    s["pul_front"] = (z, z + 3.0); z += 3.0
    s["cable_plane"] = (s["drum"][0] + s["drum"][1]) / 2
    s["end"] = z
    return s


def motor_stack(key):
    """Motor-dependent rear stack (z < 0)."""
    m = MOTORS[key]
    s = {}
    s["motor_base"] = STACK["rotor_top"] - m["h"]
    s["plate"] = (s["motor_base"] - 6.0, s["motor_base"])           # rear plate (stator mount)
    zb = s["plate"][0]
    s["board"] = (zb - 2.0 - DRIVER["board"][2], zb - 2.0)          # driver, chip facing the motor
    s["lid"] = (s["board"][0] - 5.0, s["board"][0] - 1.0)           # rear lid spigot
    s["back"] = s["lid"][0] - 1.6                                    # outer face of the lid
    s["bore_motor"] = m["od"] / 2 + 0.8
    s["bore_board"] = max(s["bore_motor"], math.hypot(*DRIVER["board"][:2]) / 2 + 0.8)
    s["outer"] = max(31.0, s["bore_motor"] + 2.0, s["bore_board"] + 2.5)
    return s


STACK = actuator_stack()

# Joint-plane mapping: |y| = CHEEK_OUT - (z_act - mount_face)
CHEEK_OUT = ARM["interior_half"] + ARM["cheek_t"]           # 22
def act_to_abs_y(z):
    return CHEEK_OUT - (z - STACK["mount_face"])
SECTOR_Y = (act_to_abs_y(STACK["cable_plane"]) - CAP["sector_t"] / 2,
            act_to_abs_y(STACK["cable_plane"]) + CAP["sector_t"] / 2)

HOUSING_R = 31.0      # front-cap outer radius (what the fork cheek supports)
CAP_SCREW_R = 27.5    # cap/mounting screw circle
TUBE_R = 19.5         # front bearing tube outer radius

# Encoder mounting (same pattern everywhere)
ENC = dict(
    board=(30.0, 18.0, 1.6),    # AS5047P breakout on the joint (edit to suit your board)
    board_holes=24.0,           # M2 hole spacing on the board
    mount_holes=36.0,           # M3 spacing of the printed bracket feet
    magnet=(6.0, 2.5),          # diametric magnet d x h
    gap=1.5,                    # magnet face -> chip
    chip_t=1.0,
)

def joint_motor(jkey):
    return MOTORS[JOINTS[jkey]["motor"]]
