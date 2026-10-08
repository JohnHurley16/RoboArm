# Inverse kinematics

`capstan_ik.py` is a single file that needs only numpy. Copy it to whatever computer drives the arm.

You describe where the gripper should be with six numbers. It gives back all five joint angles, the moteus position for each motor, and the gripper servo angle.

| Input | Meaning |
|---|---|
| `x, y, z` | Tool point, in mm. Measured from the centre of the base's underside, with z up and x forward when J1 = 0. The tool point is the middle of the Fin Ray pads, or the pad tips with `Arm(tcp="tip")`. |
| `pitch` | Angle of the gripper above horizontal, in degrees. 0 points straight out and -90 straight down. Past -90 it tilts back toward the base. |
| `roll` | Angle of the jaws about the gripper's axis, in degrees. At 0 the fingers are above and below each other; at 90 they pinch side to side. The jaws are symmetric, so roll ± 180 grips the same way, and the solver picks whichever keeps J5 in range. |
| `grip` | Gap between the pads, 1–38 mm. |

```python
from capstan_ik import Arm
arm = Arm()
sol = arm.ik(300, 0, 60, pitch=-60, roll=0, grip=20)
sol.q        # {'J1': 0.0, 'J2': 59.34, 'J3': -80.49, 'J4': -38.85, 'J5': 0.0, 'G': 23.32}
sol.motors   # {'J1': 0.0, 'J2': -1.7144, 'J3': 1.7887, 'J4': 0.7015, 'J5': 0.0, 'servo': 113.3}
sol.ok, sol.problems
arm.fk(sol.q)          # back to x, y, z, pitch, roll, grip
```

The same thing from the command line, where `--all` shows every way to reach the point:

```
python capstan_ik.py 300 0 60 --pitch -60 --grip 20 --all
```

## How it works

The arm's joint axes all line up in one vertical plane, with no sideways offsets, so the IK is solved exactly in closed form rather than iteratively.

1. **J1** points that plane at the target.
2. **Tool length:** the distance from the wrist to the tool point depends on the grip, because the parallel jaws slide forward as they close. The solver accounts for that, then steps back from the target along the pitch to find the wrist.
3. **J2 and J3** are a two-link triangle solve that places the wrist.
4. **J4** makes up the remaining pitch.
5. **J5** sets the roll.

There are up to four solutions: front or reaching back over the top, each with the elbow up or down. Each one is checked against the joint limits and against simple keep-out zones around the base, the turret and the table. Valid solutions come first. Within those it prefers front, elbow-up, unless you pass `seed=` with the current joint angles; then it picks the one closest to where the arm already is, so moves stay on the same branch.

A 5-axis arm can't reach every orientation: the gripper always points within the vertical plane through the base axis. `ik_pose(T)` takes a full 4×4 target, does the best it can, and returns the orientation error.

## Also in the file

- `jacobian(q)` and `joint_speeds(q, v, w)` convert a tool velocity into joint speeds and motor speeds. They use a damped solve, so they stay sensible near singularities.
- `line(start, end, steps)` plans a straight-line move with easing at both ends. It raises an error if any step leaves the limits or switches branch.
- `to_motors(q)` and `from_motors(m)` convert between joint angles and moteus rotor revolutions.
- `manipulability(q)` scores how far a pose is from a singularity.

## Workspace

The tables use a 20 mm grip with the tool point at the pad centre. They show the radius from the base axis at which a valid solution exists.

| Tool height | Straight down (-90°) | -75° | -60° | -45° | -30° | Level (0°) |
|---|---|---|---|---|---|---|
| z 50 mm | 132–242 | 186–316 | 248–372 | 302–420 | 340–456 | 400–454 |
| z 100 mm | 132–140 | 130–280 | 208–354 | 280–408 | 332–452 | 384–478 |

- **Straight down only works over a narrow band.** J4 stops at -90°, so pointing straight down needs the forearm level or higher. Tilting to -60° to -75° gives a much wider band. That's the natural way for this arm to pick things up.
- **Lowest pick height:** at z = 20 mm the pad tips would hit the table, which sits 22 mm past the pad centre. Pick at z ≈ 30 mm or more, or use `Arm(tcp="tip")` and aim the tips about 8 mm above the table.

## Before the first move

1. **Motor direction.** In `MOTOR_SIGN`, -1 assumes the pulley turns opposite to the drum, which is what an open capstan does. Whether it holds depends on how each motor is mounted and wired. Jog each joint slowly and flip any that move the wrong way.
2. **Zero offsets.** Put each joint at 0° (arm straight up has J2 = 90°, so zero J2 with the upper arm level). Read the moteus position and store it in `MOTOR_OFFSET`. The joint encoders on J1–J4 can do this automatically later.
3. **Gripper.** The servo sits at 90° with the jaws closed, and `SERVO_SIGN` sets which way it opens.
4. **Safety.** Use low current limits in the moteus config while testing. The keep-out zones are rough, so don't rely on them alone.

`drive_example.py` shows the whole chain on the real arm. It reads the current motor positions, solves the IK seeded from them, and sends all five moteus position commands at once with velocity, acceleration and torque limits. I haven't run it on hardware yet.

## Tests

From the repo root, run `python ik/test_ik.py`. It checks:

- every dimension, ratio and limit in the file against the CAD model;
- the frames against the CAD's own kinematics over 500 random poses (they agree to 1e-13 mm);
- forward → inverse → forward round trips on about 2,600 random reachable poses, where the original joint angles must come back as the best solution;
- that motor conversion inverts, and the Jacobian and straight-line planner.

If you change the arm's geometry, the CAD check will fail until `GEOM` here is updated to match.
