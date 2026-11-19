"""
Verify that every code cell in notebooks/Solar_Anomaly_Detection.ipynb executes successfully.
"""

import json
import sys
import os

# Use non-interactive backend so plt.show() does not block in terminal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.show = lambda *args, **kwargs: None

notebook_path = os.path.join("notebooks", "Solar_Anomaly_Detection.ipynb")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Global execution namespace
nb_globals = {
    '__file__': os.path.abspath(notebook_path),
    'display': print,
    'plt': plt
}

code_cells = [cell for cell in nb["cells"] if cell["cell_type"] == "code"]
print(f"Total Code Cells to Execute: {len(code_cells)}", flush=True)

for idx, cell in enumerate(code_cells, start=1):
    source = "".join(cell["source"])
    first_line = source.strip().split("\n")[0] if source.strip() else "(empty)"
    print(f"\n[Cell {idx}/{len(code_cells)}] {first_line[:65]}...", flush=True)
    try:
        exec(source, nb_globals)
        print(f"✓ Cell {idx} completed successfully.", flush=True)
    except Exception as e:
        print(f"✗ ERROR in Cell {idx}: {e}", flush=True)
        sys.exit(1)

print("\n" + "=" * 80, flush=True)
print(" ALL 16 NOTEBOOK SECTIONS EXECUTED FLAWLESSLY FROM FIRST CELL TO LAST CELL!", flush=True)
print("=" * 80, flush=True)
