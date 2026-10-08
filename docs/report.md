# Design report

Drive: gimbal motor → zero-backlash TPU jaw coupling → capstan (no gearbox). Each joint has a mjbots moteus-c1 driver on the back of its motor, reading the motor's own encoder and the joint encoder on its aux port.

| Joint | Motor | Capstan ratio | Rated / peak at the joint (η≈0.92) | Gravity, worst pose (0 / 200 / 500 g payload) | Range of motion | Cable |
|---|---|---:|---:|---:|---:|---:|
| J1 base yaw | CubeMars GL40 KV70 | 9.7:1 | 2.23 / 4.46 Nm | – | -150° … 150° | 501 mm |
| J2 shoulder | CubeMars GL60 KV25 | 10.4:1 | 5.74 / 16.74 Nm | 3.20 / 4.15 / 5.56 Nm | -10° … 120° | 367 mm |
| J3 elbow | CubeMars GL60 KV25 | 8.0:1 | 4.42 / 12.88 Nm | 1.34 / 1.99 / 2.98 Nm | -110° … 110° | 396 mm |
| J4 wrist pitch | CubeMars GL40 KV70 | 6.5:1 | 1.50 / 2.99 Nm | 0.39 / 0.78 / 1.38 Nm | -90° … 70° | 329 mm |
| J5 wrist roll | CubeMars GL40 KV70 | 6.0:1 | 1.38 / 2.76 Nm | – | -150° … 150° | 394 mm |

Moving mass (everything past the shoulder): 1497 g. The shoulder gravity figures are without the balancer.

Shoulder height: 155 mm. Upper arm 150 mm, forearm 130 mm, wrist pitch axis to fingertip ≈ 228 mm.

Gripper: MG90S metal-gear micro servo driving a synced parallelogram. Pad gap 1.0 mm (closed) to 37.8 mm (open), before Fin Ray deflection. Squeeze at stall ≈ 8.8 N per finger (servo stall torque / link length).

## Shoulder balancer

Zero-length cable spring: ear on the upper arm → idler on the turret → 2:1 spring cartridge behind the turret.

| | Without balancer | With balancer |
|---|---:|---:|
| Peak shoulder torque (any pose) | 3.21 Nm | 1.71 Nm |
| RMS shoulder torque (over all poses) | 1.79 Nm | 0.81 Nm |
| Motor heating (∝ torque², RMS) | 1× | 0.21× |

Cable tension 11–70 N over the shoulder's range (1 mm Dyneema ≈ 1 kN break).

Spring: music wire 2.2 mm, OD 21 mm, free length ≈ 78 mm, rate ≈ 3.2 N/mm, 12.7 total coils, ends closed & ground. Working stress 732 MPa (limit 789). Deflection 6.9–42.9 mm, solid length 28.0 mm.

The ear has three pin holes (44 / 47 / 50 mm). Middle is nominal; move the pin out if the arm sags, in if it floats up. Each hole step is about ±6 % balance.

## Position accuracy

There is no gearbox, so there is no gear backlash. The error budget is coupling slack (zero when new), cable stretch under load, and the encoders.

| Joint | Cable stiffness | Stretch under worst gravity load | Coupling slack if the spider takes a 0.05 mm set | Motor-encoder step at the joint |
|---|---:|---:|---:|---:|
| J1 | 1648 Nm/rad | 0.000° | 0.056° | 0.0023° |
| J2 | 1844 Nm/rad | 0.053° | 0.052° | 0.0021° |
| J3 | 1205 Nm/rad | 0.064° | 0.068° | 0.0027° |
| J4 | 858 Nm/rad | 0.026° | 0.084° | 0.0034° |
| J5 | 752 Nm/rad | 0.000° | 0.091° | 0.0037° |

| Pose | Reach | Backlash (new) | Slack if worn (worst / RSS) | Cable stretch | Motor-encoder step |
|---|---:|---:|---:|---:|---:|
| Reach (arm straight out) | 485 mm | 0 | ±0.81 / ±0.41 mm | 0.91 mm | 0.065 mm |
| Table pick (elbow down) | 188 mm | 0 | ±0.57 / ±0.29 mm | 0.73 mm | 0.046 mm |
| Close in (folded) | 209 mm | 0 | ±0.55 / ±0.28 mm | 0.66 mm | 0.044 mm |

Controlling J1–J4 on their joint encoders removes coupling slack and cable stretch from the static position. The error is then set by the joint encoders (arm straight out, worst / RSS):

- 14-bit resolution: ±0.6 / ±0.3 mm (repeatability)
- 0.1° after calibration: ±2.6 / ±1.4 mm
- 0.3° uncalibrated: ±7.9 / ±4.1 mm

Cable stiffness assumes an effective EA of 15 kN for 1 mm Dyneema braid (an estimate; real braid varies and creeps).

