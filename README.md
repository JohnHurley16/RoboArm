# Capstan Arm 5

A 3D-printed 5-axis robot arm, built around zero-backlash capstan drives, plus a GPU simulator for teaching it things like cornhole with reinforcement learning.

![Capstan Arm 5](docs/images/final_pose.png)

- **Drive:** every joint is a CubeMars gimbal motor driving a Dyneema capstan directly. There's no gearbox, so it's backdrivable, quiet, and torque can be sensed from motor current.
- **Electronics:** a moteus-c1 driver on every motor, plus joint encoders on J1–J4.
- **Shoulder:** a spring balancer carries about half the gravity load.
- **Gripper:** parallel-jaw, with Fin Ray TPU fingertips.
- **Printing:** every printed part fits a Bambu Lab P1S. The plates are laid out in assembly order, so you can build while the rest prints.
- **Control:** closed-form inverse kinematics (numpy only) gives you moteus commands from a tool pose.
- **Simulator:** MuJoCo/MJX, generated from the CAD, with real motor curves, a floppy beanbag and a real table pickup. Several RL algorithms can train in parallel.

## Repository layout

| Folder | What's in it |
|---|---|
| [`cad/`](cad) | The parametric CAD (CadQuery). `cad/arm/params.py` holds every dimension. `build.py` regenerates everything; `analysis_*.py` covers torque, balancer and accuracy; `check_*.py` runs the collision and range-of-motion checks; `make_plates.py` builds the print plates and `make_mini.py` the desk model. |
| [`print/`](print) | Ready to print: `stl/` (every part, in print orientation), `plates/` (P1S plates in assembly order, with `PLATES.md`), `mini/` (multi-colour desk model), `step/` (assembly and actuators for other CAD tools). |
| [`docs/`](docs) | [`DESIGN.md`](docs/DESIGN.md) (design, assembly, wiring, balancer, accuracy), `report.md` (torque budget), `BOM.md` and renders. |
| [`bom/`](bom) | Priced shopping list (`capstan_arm_parts.html`, `bom_priced.csv`). Edit `bom_data.py`, then run `make_bom.py`. |
| [`viewer/`](viewer) | Interactive 3D viewer (three.js). Serve the folder and open `index.html`. |
| [`ik/`](ik) | Inverse kinematics: tool pose → joint angles → moteus positions. See [`IK.md`](ik/IK.md). |
| [`sim/`](sim) | GPU simulator, training environments and training launcher. See [`SIM.md`](sim/SIM.md). |
| [`legacy/`](legacy) | The first design: drone motors driving a cycloid then a capstan. Reference only. |

## Quick start

```bash
# CAD: regenerate everything after editing cad/arm/params.py (about 3 min)
pip install -r requirements/cad.txt
python cad/build.py
python cad/check_interf.py home && python cad/check_poses.py && python cad/check_balancer.py
python cad/make_plates.py && python cad/export_print_files.py

# Inverse kinematics (numpy only)
python ik/capstan_ik.py 300 0 60 --pitch -60 --grip 20 --all

# Simulator and training (Linux/WSL2 with an NVIDIA GPU)
pip install -r requirements/sim.txt
python sim/smoke_test.py
python sim/train.py --population sim/populations/throw.json
python sim/leaderboard.py runs/throw
```

## Before you build

- **Motor hole patterns:** the motor and moteus mounting-hole patterns in `cad/arm/params.py` are placeholders. Copy them from the CubeMars and mjbots drawings, then rebuild.
- **First power-up:** check each joint's direction and zero (`MOTOR_SIGN` and `MOTOR_OFFSET` in `ik/capstan_ik.py`) with low current limits.
- **Simulator tuning:** the beanbag's floppiness and the pinch grip strength are estimates. `sim/SIM.md` explains how to measure your real bag and gripper and update them.
