"""Copy the generated print files from cad/out/ into print/ (the committed copies).

    python cad/build.py && python cad/make_plates.py && python cad/make_mini.py 1/4.2
    python cad/export_print_files.py
"""
import glob
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
PRINT = os.path.join(os.path.dirname(HERE), "print")
DOCS = os.path.join(os.path.dirname(HERE), "docs")

COPY = [
    ("stl/*.stl", "stl"),
    ("plates/*", "plates"),
    ("mini/capstan_arm_mini.3mf", "mini"), ("mini/MINI.md", "mini"), ("mini/*.png", "mini"),
    ("step/arm_assembly.step", "step"), ("step/actuator_GL40.step", "step"), ("step/actuator_GL60.step", "step"),
]

for pattern, dest in COPY:
    os.makedirs(os.path.join(PRINT, dest), exist_ok=True)
    for f in glob.glob(os.path.join(OUT, pattern)):
        if os.path.isfile(f):
            shutil.copy(f, os.path.join(PRINT, dest))
for f in ("report.md", "BOM.md", "BOM.csv"):
    if os.path.exists(os.path.join(OUT, f)):
        shutil.copy(os.path.join(OUT, f), DOCS)
print("print/ and docs/ updated from cad/out/")
