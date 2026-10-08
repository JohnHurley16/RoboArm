# Print files

| Folder | What's in it |
|---|---|
| `stl/` | Every printed part, already rotated to its print orientation. |
| `plates/` | Bambu Lab P1S plates in assembly order (`plate_NN_sX_colour.3mf`), plus `PLATES.md`, which has the timeline and what to build while each stage prints. |
| `mini/` | 3 in tall multi-colour desk model: one plate, AMS slots 1 white, 2 graphite, 3 orange, 4 black. See `MINI.md`. |
| `step/` | Full assembly and both actuator sizes, for use in other CAD tools. |

These files are generated. After a CAD change, run `python cad/build.py`, `python cad/make_plates.py` and `python cad/make_mini.py 1/4.2`, then `python cad/export_print_files.py` to update them here.

**Materials:** PETG in white, graphite and orange; TPU 95A for the spiders and Fin Ray tips. The design guide (`docs/DESIGN.md`, Printing section) covers infill and supports.
