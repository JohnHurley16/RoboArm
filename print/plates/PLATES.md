# Print plates in assembly order (Bambu Lab P1S, 256 × 256 mm)

The plates are numbered in the order to print them. They're grouped into **stages**: when a stage's last plate comes off the printer, start the next plate and build that stage while it prints. Each stage's build takes a few hours, and the next stage's plates usually take longer than that, so the printer is rarely waiting on you.

The most technical work comes first: the actuators. Stage 1 is one complete actuator, so you can check bearing fits, the coupling and the driver setup on a single unit before printing the other four.

Each plate is one filament. Some plates carry a few small parts from the next stage in their spare room (marked *early*), so nothing arrives late and no near-empty plates are needed.

Open each `plate_NN_*.3mf` in Bambu Studio. The parts keep their positions and orientations. Pick the filament named in the table, slice, print.

## Timeline

| Stage | Plates | Printer time | Build while the next stage prints |
|---:|---|---:|---|
| 1 | 1–3 | ≈ 8 h (done at ≈ 8 h) | Pathfinder actuator: J2 shoulder (GL60) |
| 2 | 4–6 | ≈ 27 h (done at ≈ 35 h) | The other four actuators: J3 (GL60), J1, J4, J5 (GL40) |
| 3 | 7–8 | ≈ 41 h (done at ≈ 76 h) | Base and turret (J1 yaw) |
| 4 | 9–11 | ≈ 18 h (done at ≈ 94 h) | Upper arm and shoulder balancer (J2) |
| 5 | 12 | ≈ 11 h (done at ≈ 105 h) | Forearm (J3 elbow) |
| 6 | 13–14 | ≈ 5 h (done at ≈ 110 h) | Wrist and gripper (J4 pitch, J5 roll) |

Total ≈ 1487 g of filament and ≈ 110 h of printing. The gram and hour figures are rough (about 60 % effective fill, about 14 g/h, plus heat-up); Bambu Studio's slice preview gives the real numbers.

## Stage 1: Pathfinder actuator: J2 shoulder (GL60)

| Plate | File | Filament | Supports | ≈ g | ≈ h | Parts |
|---:|---|---|---|---:|---:|---|
| 1 | `plate_01_s1_orange_supports.3mf` | PETG, orange | tree (auto) | 13 | 1.2 | act_bearing_spacer, act_hub, act_magnet_plug_GL60, act_retainer_ring, act_rotor_adapter_GL60 |
| 2 | `plate_02_s1_graphite.3mf` | PETG, graphite / dark grey | none | 73 | 5.5 | act_front_cap, act_motor_can_GL60, act_rear_lid_GL60, act_rear_plate_GL60 |
| 3 | `plate_03_s1_tpu.3mf` | TPU 95A, black (external spool, not the AMS) | none | 19 | 1.7 | act_spider_tpu ×5, finray_tpu ×2 |

All the TPU is on this stage, so the external spool only has to be loaded once.

**Fit check, before starting plate 4:** This is the one time to let the printer sit idle for an hour. Before you start plate 4, press both 6705 bearings into the front cap and slide the hub through them. They should go in with firm thumb pressure, and the hub should spin with no rock. The spider should need a light push to seat on the dogs. If anything is loose or won't go in, adjust `FIT` in `arm/params.py`, run `python build.py` and `python cad/make_plates.py`, and reprint plates 1–2 before going on. Every other actuator part uses the same fits.

**While stage 2 prints:**

1. Finish the actuator (README, *Actuator*, steps 1–6).
2. Wire its moteus-c1 to the 24 V supply and the CAN-FD adapter. Set its CAN ID to 2 (`conf set id.id 2`, then `conf write`).
3. Calibrate it with the motor free to spin (`moteus_tool -t 2 --calibrate`), before any cable is wound.
4. Spin it in position mode, then turn the hub by hand with the power off. It should be smooth, with no click at reversal. Check that the motor-encoder and aux-encoder readings agree.

## Stage 2: The other four actuators: J3 (GL60), J1, J4, J5 (GL40)

| Plate | File | Filament | Supports | ≈ g | ≈ h | Parts |
|---:|---|---|---|---:|---:|---|
| 4 | `plate_04_s2_orange_supports.3mf` | PETG, orange | tree (auto) | 81 | 6.1 | act_bearing_spacer ×4, act_hub ×4, act_magnet_plug_GL40 ×3, act_magnet_plug_GL60, act_retainer_ring ×4, act_rotor_adapter_GL40 ×3, act_rotor_adapter_GL60; *early, for stage 3:* turret_drum |
| 5 | `plate_05_s2_graphite.3mf` | PETG, graphite / dark grey | none | 156 | 11.5 | act_front_cap, act_motor_can_GL40 ×2, act_motor_can_GL60, act_rear_lid_GL60, act_rear_plate_GL60 |
| 6 | `plate_06_s2_graphite.3mf` | PETG, graphite / dark grey | none | 123 | 9.1 | act_front_cap ×3, act_motor_can_GL40, act_rear_lid_GL40 ×3, act_rear_plate_GL40 ×3; *early, for stage 3:* j1_spacer, joint_cap |

**While stage 3 prints:**

1. Build the remaining four actuators exactly like the first. Label them: the GL60 is J3; the GL40s are J1, J4 and J5.
2. Set CAN IDs 1, 3, 4 and 5, then calibrate each one free-spinning.
3. Daisy-chain all five drivers on one CAN bus and confirm every ID answers.

## Stage 3: Base and turret (J1 yaw)

| Plate | File | Filament | Supports | ≈ g | ≈ h | Parts |
|---:|---|---|---|---:|---:|---|
| 7 | `plate_07_s3_graphite.3mf` | PETG, graphite / dark grey | none | 368 | 26.6 | base |
| 8 | `plate_08_s3_white.3mf` | PETG, white | none | 192 | 14.0 | joint_cap, turret_body |

**While stage 4 prints:**

1. Fit the heat-set inserts in the base and turret.
2. Press the 608s into the base boss, mount the J1 actuator, and fit the J1 encoder and its graphite joint cap under the base.
3. Fit the 8 mm shaft, `j1_spacer`, `turret_drum` and `turret_body`. Wind and tension the J1 cable.
4. Mount the J2 actuator in the turret cheek and fit the J2 encoder with its white joint cap.
5. Test J1: sweep the turret through its range under power and check the cable doesn't ride up the drum.

## Stage 4: Upper arm and shoulder balancer (J2)

| Plate | File | Filament | Supports | ≈ g | ≈ h | Parts |
|---:|---|---|---|---:|---:|---|
| 9 | `plate_09_s4_orange.3mf` | PETG, orange | none | 26 | 2.1 | balancer_idler, balancer_piston, balancer_spool, sector_j2; *early, for stage 5:* sector_j3 |
| 10 | `plate_10_s4_graphite.3mf` | PETG, graphite / dark grey | none | 121 | 8.9 | balancer_bracket, balancer_piston_pulley, balancer_spacer, upper_arm_cheek_a, upper_arm_cheek_b; *early, for stage 5:* forearm_cheek_a, forearm_cheek_b |
| 11 | `plate_11_s4_white.3mf` | PETG, white | none | 98 | 7.3 | balancer_tube, joint_cap, upper_arm; *early, for stage 5:* joint_cap |

**While stage 5 prints:**

1. Bolt `sector_j2` to the upper arm and fit the J2 shaft into the turret.
2. Wind and tension the J2 cable. Test the shoulder through its range with a low current limit.
3. Bolt on both elbow cheeks with the J3 actuator, and fit the J3 encoder with its joint cap.
4. Build and fit the balancer (README, *Shoulder balancer*), then do its motors-off drift test.

## Stage 5: Forearm (J3 elbow)

| Plate | File | Filament | Supports | ≈ g | ≈ h | Parts |
|---:|---|---|---|---:|---:|---|
| 12 | `plate_12_s5_white.3mf` | PETG, white | none | 154 | 11.3 | forearm; *early, for stage 6:* finger ×2, hand, palm, servo_cover |

**While stage 6 prints:**

1. Bolt `sector_j3` to the forearm and fit it into the elbow.
2. Wind and tension the J3 cable and test the elbow.
3. Bolt on the wrist cheeks with the J4 actuator, and fit the J4 encoder with its joint cap.

## Stage 6: Wrist and gripper (J4 pitch, J5 roll)

| Plate | File | Filament | Supports | ≈ g | ≈ h | Parts |
|---:|---|---|---|---:|---:|---|
| 13 | `plate_13_s6_orange.3mf` | PETG, orange | none | 32 | 2.6 | roll_flange, sector_j4 |
| 14 | `plate_14_s6_graphite.3mf` | PETG, graphite / dark grey | none | 31 | 2.5 | link_drive_lower, link_drive_upper, link_follow ×2, roll_pulley_guard, roll_spacer, wrist_yoke |

**Once the last plate is off:**

1. Bolt `sector_j4` to the hand, fit it into the wrist and wind the J4 cable.
2. Build the wrist roll and gripper (README, *Wrist roll* and *Gripper*), using the Fin Ray tips from stage 1.
3. Full-arm test: home every joint, then run the range-of-motion sweep slowly with low current limits.

## Settings

- PETG: textured PEI plate, 0.2 mm layers, 4 walls, 30 % gyroid. Open the lid or remove the top glass.
- 100 % infill for `act_hub` and `act_rotor_adapter_*`: select them and set infill per object.
- Plates marked 'tree (auto)': enable tree supports (auto). Only the `act_hub` flanges need them.
- `base` is the longest print, about a day at these settings. 3 walls and 15 % gyroid are plenty for it and save several hours. Start it in the evening.
- TPU 95A: feed from the external spool holder (not the AMS), about 3–4 mm³/s. Fin Ray tips: 2 walls, 0 % infill. Spiders: 100 % infill.
- Install heat-set inserts after printing, with the parts cool.
- To change the order, edit `STAGES` in `cad/make_plates.py` and rerun it. It refuses to run if a printed part is missing from the stages or listed twice.

